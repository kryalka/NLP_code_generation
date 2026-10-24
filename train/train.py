import argparse
from contextlib import closing
from pathlib import Path
from math import cos, pi
from shutil import copyfile
from time import perf_counter
import random
import signal

import yaml

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.baseline import Transformer
from huggingface_hub import hf_hub_download
from models.code_format import HEADER, restore_solve_wrapper
from models.hf_data import add_token_ids, read_frame, resolve_revision
from models.tokenization import file_hash, tokenize_file
from models.utils import ROOT, comet_run, load_tokenizer
from train.checkpoints import (
    random_state, restore_checkpoint, restore_random_state, save_checkpoint,
    save_epoch_checkpoint, training_contract, upload_checkpoint,
)
from train.batches import prefetch_from_cursor
from val.val import evaluate, reference_answers


pause_requested = False


def set_learning_rate(optimizer, settings, step, total_steps):
    rate = settings["learning_rate"]
    schedule = settings.get("lr_schedule", "constant")
    if schedule in ("linear", "cosine"):
        warmup = settings["warmup_steps"]
        if step <= warmup:
            rate *= step / warmup
        elif schedule == "linear":
            rate *= max(0.0, (total_steps - step) / (total_steps - warmup))
        else:
            progress = min(1.0, (step - warmup) / (total_steps - warmup))
            rate *= 0.5 * (1.0 + cos(pi * progress))
    elif schedule != "constant":
        raise ValueError(f"Неизвестный lr_schedule: {schedule}")
    for group in optimizer.param_groups:
        group["lr"] = rate


def optimizer_step(model, optimizer, scaler, gradient_multiplier, settings):
    scaler.unscale_(optimizer)
    if gradient_multiplier != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_multiplier)
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return scaler.get_scale() >= old_scale


def request_pause(signum, frame):
    global pause_requested
    pause_requested = True
    print("Сохраняемся после накопления градиентов", flush=True)


def train(cfg, experiment, saved=None, initial=None):
    settings, model_cfg = cfg["training"], cfg["model"]
    max_steps = settings["max_steps"]
    max_seconds = settings.get("max_seconds")
    deadline = perf_counter() + max_seconds if max_seconds is not None else None
    stop_after = settings.get("stop_after_steps")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.use_deterministic_algorithms(settings.get("deterministic", False))
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = training_contract(cfg)

    model = Transformer(model_cfg).to(device)
    if initial is not None:
        try:
            model.load_state_dict(initial["model"], strict=True)
        except RuntimeError as error:
            raise ValueError("Архитектура initial checkpoint отличается от конфига") from error
        print({"initialized_from_step": initial.get("step"),
               "initialized_from": cfg["initial_weights"]["filename"]}, flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"], fused=device.type == "cuda",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    training_model = torch.compile(model, dynamic=True) if settings.get("compile", False) else model
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / cfg["tokenizer_path"]
    if tokenizer_path.resolve() != (output_dir / "tokenizer.json").resolve():
        copyfile(tokenizer_path, output_dir / "tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path, model_cfg["vocab_size"])

    preparation_started = perf_counter()
    hf = cfg.get("hf_data", {})
    if hf.get("enabled", False):
        revision = resolve_revision(hf["repo_id"], hf.get("revision", "main"))
        local_train_tokens = hf.get("local_train_tokens")
        if local_train_tokens and Path(local_train_tokens).is_file():
            train_tokens = local_train_tokens
        else:
            train_tokens = hf_hub_download(
                repo_id=hf["repo_id"],
                filename=hf["train_tokens_file"],
                repo_type="dataset",
                revision=revision,
                cache_dir=ROOT / "data/hf_cache",
            )
        source = pl.scan_parquet(train_tokens).drop_nulls()
        val = read_frame(
            hf["repo_id"], hf["val_file"], revision,
            ["question_clean", "code_clean"],
        )
        if hf.get("restore_wrappers", False):
            val = val.with_columns(
                pl.col("code_clean").map_elements(
                    restore_solve_wrapper, return_dtype=pl.String
                )
            )
        val = add_token_ids(val, tokenizer)
        data_signature = {
            "hf_repo": hf["repo_id"], "hf_revision": revision,
            "train_tokens_file": hf["train_tokens_file"], "val_file": hf["val_file"],
            "tokenizer": file_hash(tokenizer_path),
        }
    else:
        train_tokens = tokenize_file(ROOT / cfg["train_path"], tokenizer_path, cfg)
        val_tokens = tokenize_file(ROOT / cfg["val_path"], tokenizer_path, cfg)
        data_signature = {
            "train": train_tokens.name, "val": val_tokens.name,
            "tokenizer": file_hash(tokenizer_path),
        }
        source = pl.scan_parquet(train_tokens).drop_nulls()
        val = pl.read_parquet(ROOT / cfg["val_path"], columns=["question_clean", "code_clean"])
        val = val.hstack(pl.read_parquet(val_tokens))
    target_length = pl.col("code_ids").list.len()
    first_code_ids = source.select("code_ids").head(1).collect().item()
    first_target = tokenizer.decode(first_code_ids)
    expects_body = model_cfg.get("body_only_targets", False)
    if expects_body == first_target.startswith(HEADER):
        expected = "тело функции" if expects_body else "полная функция def solve()"
        raise ValueError(f"Неверный формат train targets: ожидалось {expected}")
    valid_target = target_length <= model_cfg["max_target_length"]
    dataset_stats = source.select(
        pl.len().alias("rows"),
        pl.when(valid_target).then(target_length).otherwise(None).mean().alias("mean_target_tokens"),
    ).collect(engine="streaming").row(0, named=True)
    total_rows = dataset_stats["rows"]
    configured_tokens = settings.get("tokens_per_update")
    if configured_tokens == "auto":
        reference_examples = settings.get(
            "reference_examples_per_update",
            settings["batch_size"] * settings["gradient_accumulation_steps"],
        )
        tokens_per_update = round(reference_examples * dataset_stats["mean_target_tokens"])
    elif configured_tokens is None:
        tokens_per_update = None
        reference_examples = settings["batch_size"] * settings["gradient_accumulation_steps"]
    else:
        tokens_per_update = int(configured_tokens)
        reference_examples = settings.get("reference_examples_per_update")
    if saved is not None:
        if saved.get("format_version") != 1:
            raise ValueError("Неподдерживаемый формат checkpoint")
        saved_contract = dict(saved["contract"])
        saved_contract["training"] = {
            key: value for key, value in saved_contract["training"].items()
            if key != "pass_every_steps"
        }
        if saved_contract != contract:
            raise ValueError("Параметры обучения отличаются от checkpoint")
        if saved["data_signature"] != data_signature:
            raise ValueError("Данные или токенизатор отличаются от checkpoint")
    answers = reference_answers(val, model_cfg.get("body_only_targets", False))

    state = {
        "step": 0, "attempts": 0, "cursor": {"epoch": 0, "chunk": 0, "batch": 0},
        "seen_rows": 0, "epoch_rows": 0, "window_tokens": 0, "window_loss": 0.0,
        "epoch_tokens": 0, "epoch_loss": 0.0, "train_seconds": 0.0,
        "best_val_loss": float("inf"), "best_pass_at_1": -1.0,
        "metrics": {}, "metrics_step": -1,
        "window_padded_tokens": 0, "window_sequence_tokens": 0,
        "window_update_examples": 0, "window_update_tokens": 0, "window_updates": 0,
    }
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["training_state"]
        state.setdefault("window_padded_tokens", 0)
        state.setdefault("window_sequence_tokens", 0)
        state.setdefault("window_update_examples", 0)
        state.setdefault("window_update_tokens", 0)
        state.setdefault("window_updates", 0)
        state.setdefault("best_pass_at_1", -1.0)
        restore_random_state(saved["rng"])
        print({"resumed_step": state["step"], "cursor": state["cursor"],
               "learning_rate": optimizer.param_groups[0]["lr"]}, flush=True)
    if experiment is not None:
        experiment.log_metrics({
            "preparation_seconds": perf_counter() - preparation_started,
            "mean_target_tokens": dataset_stats["mean_target_tokens"],
            "reference_examples_per_update": reference_examples,
            "configured_target_tokens_per_update": tokens_per_update or 0,
        }, step=state["step"])
    print({
        "total_steps": max_steps, "warmup_steps": settings["warmup_steps"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "mean_target_tokens": dataset_stats["mean_target_tokens"],
        "reference_examples_per_update": reference_examples,
        "target_tokens_per_update": tokens_per_update,
    }, flush=True)
    window_loss = torch.tensor(state["window_loss"], dtype=torch.float64, device=device)
    epoch_loss = torch.tensor(state["epoch_loss"], dtype=torch.float64, device=device)
    optimizer.zero_grad(set_to_none=True)
    stop_reason = "completed" if state["step"] >= max_steps else "running"
    best_checkpoint_changed = False
    best_pass_checkpoint_changed = False

    def weights():
        return {
            "model": model.state_dict(), "config": model_cfg,
            "step": state["step"], "epoch": state["cursor"]["epoch"] + int(state["epoch_rows"] > 0),
            "planned_steps": max_steps, "schedule_step": state["attempts"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metrics": state["metrics"], "metrics_step": state["metrics_step"],
            "stop_reason": stop_reason,
        }

    def save_resume():
        nonlocal best_checkpoint_changed, best_pass_checkpoint_changed
        state["window_loss"] = window_loss.item()
        state["epoch_loss"] = epoch_loss.item()
        checkpoint = weights()
        checkpoint.update(
            format_version=1, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            rng=random_state(), training_state=dict(state), contract=contract,
            data_signature=data_signature, answers=answers,
            comet_key=experiment.get_key() if experiment is not None else None,
            experiment_name=cfg["comet"]["experiment_name"],
        )
        path = output_dir / "resume.pt"
        save_checkpoint(checkpoint, path)
        if best_checkpoint_changed:
            best_checkpoint_changed = not upload_checkpoint(
                output_dir / "best_val_loss.pt", cfg
            )
        if best_pass_checkpoint_changed:
            best_pass_checkpoint_changed = not upload_checkpoint(
                output_dir / "best_pass_at_1.pt", cfg
            )
        upload_checkpoint(path, cfg)

    def validate(compute_pass=False):
        nonlocal best_checkpoint_changed, best_pass_checkpoint_changed
        rng = random_state()
        try:
            metrics = evaluate(model, tokenizer, val, answers, cfg, device, compute_pass=compute_pass)
        finally:
            restore_random_state(rng)
        state["metrics"] = metrics
        state["metrics_step"] = state["step"]
        print({"step": state["step"], **metrics}, flush=True)
        if experiment is not None:
            values = dict(metrics)
            values["loss"] = values.pop("val_loss")
            experiment.log_metrics(values, prefix="val", step=state["step"])
        if metrics["val_loss"] < state["best_val_loss"]:
            state["best_val_loss"] = metrics["val_loss"]
            save_checkpoint(weights(), output_dir / "best_val_loss.pt")
            best_checkpoint_changed = True
        if compute_pass and metrics["pass_at_1"] > state["best_pass_at_1"]:
            state["best_pass_at_1"] = metrics["pass_at_1"]
            save_checkpoint(weights(), output_dir / "best_pass_at_1.pt")
            best_pass_checkpoint_changed = True
        if compute_pass:
            path = output_dir / "model.pt"
            save_checkpoint(weights(), path)
            upload_checkpoint(path, cfg)
            if state["step"] in settings.get("snapshot_steps", []):
                snapshot = output_dir / f"model_step_{state['step']}.pt"
                save_checkpoint(weights(), snapshot)
                upload_checkpoint(snapshot, cfg)

    if saved is None:
        save_resume()
    accumulated = 0
    accumulated_tokens = 0
    accumulated_examples = 0
    while state["step"] < max_steps:
        epoch = state["cursor"]["epoch"]
        model.train()
        batches = prefetch_from_cursor(source, settings, model_cfg, dict(state["cursor"]), total_rows, device.type == "cuda")
        with closing(batches), tqdm(total=total_rows, initial=state["epoch_rows"], desc=f"Epoch {epoch + 1}", unit="строк") as progress:
            for prepared, rows, tokens, padded_tokens, sequence_tokens, cursor, epoch_end in batches:
                started = perf_counter()
                state["seen_rows"] += rows
                state["epoch_rows"] += rows
                state["cursor"] = cursor
                progress.update(rows)
                if prepared is not None:
                    src, decoder_input, labels = (value.to(device, non_blocking=True) for value in prepared)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = training_model(src, decoder_input, labels)
                        loss = F.cross_entropy(logits.float(), labels[labels.ne(-100)])
                    if tokens_per_update is None:
                        scaled_loss = loss / settings["gradient_accumulation_steps"]
                    else:
                        scaled_loss = loss * (tokens / tokens_per_update)
                    scaler.scale(scaled_loss).backward()
                    accumulated += 1
                    accumulated_tokens += tokens
                    accumulated_examples += labels.size(0)
                    window_loss.add_(loss.detach().double() * tokens)
                    epoch_loss.add_(loss.detach().double() * tokens)
                    state["window_tokens"] += tokens
                    state["epoch_tokens"] += tokens
                    state["window_padded_tokens"] += padded_tokens
                    state["window_sequence_tokens"] += sequence_tokens
                    del logits, loss, scaled_loss

                updated = False
                target_reached = (
                    accumulated == settings["gradient_accumulation_steps"]
                    if tokens_per_update is None else accumulated_tokens >= tokens_per_update
                )
                if target_reached or (epoch_end and accumulated):
                    state["attempts"] += 1
                    set_learning_rate(optimizer, settings, state["step"] + 1, max_steps)
                    if tokens_per_update is None:
                        multiplier = settings["gradient_accumulation_steps"] / accumulated
                    else:
                        multiplier = tokens_per_update / accumulated_tokens
                    updated = optimizer_step(model, optimizer, scaler, multiplier, settings)
                    state["step"] += int(updated)
                    if updated:
                        state["window_update_examples"] += accumulated_examples
                        state["window_update_tokens"] += accumulated_tokens
                        state["window_updates"] += 1
                    accumulated = 0
                    accumulated_tokens = 0
                    accumulated_examples = 0
                state["train_seconds"] += perf_counter() - started

                if updated and state["step"] % cfg["comet"]["log_every_steps"] == 0:
                    recent_loss = window_loss.item() / state["window_tokens"]
                    padding_fraction = 1.0 - state["window_sequence_tokens"] / state["window_padded_tokens"]
                    metrics = dict(loss=recent_loss, learning_rate=optimizer.param_groups[0]["lr"],
                                   seen_rows=state["seen_rows"], seconds=state["train_seconds"],
                                   skipped_updates=state["attempts"] - state["step"], planned_steps=max_steps,
                                   padding_fraction=padding_fraction,
                                   examples_per_update=state["window_update_examples"] / state["window_updates"],
                                   target_tokens_per_update=state["window_update_tokens"] / state["window_updates"])
                    print({"step": state["step"], **metrics}, flush=True)
                    if experiment is not None:
                        experiment.log_metrics(metrics, prefix="train", step=state["step"], epoch=epoch + 1)
                    window_loss.zero_()
                    state["window_tokens"] = 0
                    state["window_padded_tokens"] = 0
                    state["window_sequence_tokens"] = 0
                    state["window_update_examples"] = 0
                    state["window_update_tokens"] = 0
                    state["window_updates"] = 0

                if epoch_end:
                    if experiment is not None:
                        experiment.log_metrics({"epoch_loss": epoch_loss.item() / state["epoch_tokens"],
                                                "seconds": state["train_seconds"]}, prefix="train", step=state["step"], epoch=epoch + 1)
                    state["epoch_rows"] = 0
                    state["epoch_tokens"] = 0
                    epoch_loss.zero_()
                    path = save_epoch_checkpoint(weights(), output_dir, keep=3)
                    upload_checkpoint(path, cfg)

                if state["step"] >= max_steps:
                    stop_reason = "completed"
                elif accumulated == 0 and deadline is not None and perf_counter() >= deadline:
                    stop_reason = "time_limit"
                elif accumulated == 0 and stop_after is not None and state["step"] >= stop_after:
                    stop_reason = "paused"
                elif accumulated == 0 and pause_requested:
                    stop_reason = "paused"
                if stop_reason != "running":
                    break
                compute_pass = updated and state["step"] % settings["pass_every_steps"] == 0
                if updated and (state["step"] % settings["val_every_steps"] == 0 or compute_pass):
                    validate(compute_pass)
                if updated and state["step"] % settings["checkpoint_every_steps"] == 0:
                    save_resume()
                    save_checkpoint(weights(), output_dir / "last_weights.pt")
        if stop_reason != "running":
            break

    save_resume()
    save_checkpoint(weights(), output_dir / "last_weights.pt")

    if stop_reason == "completed":
        validate(compute_pass=True)
        save_resume()
    print({"step": state["step"], "stop_reason": stop_reason, "resume": str(output_dir / "resume.pt")}, flush=True)
    if experiment is not None:
        experiment.log_metrics({"completed": int(stop_reason == "completed"), "last_step": state["step"],
                                "best_val_loss": state["best_val_loss"],
                                "best_pass_at_1": state["best_pass_at_1"]}, step=state["step"])
    return state


def main():
    global pause_requested
    pause_requested = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/m1_deep_6e6d_d384_100k.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    output_dir = ROOT / cfg["output_dir"]
    resume_from = cfg["training"].get("resume_from")
    resume_path = ROOT / resume_from if resume_from else output_dir / "resume.pt"
    if resume_from and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    if not resume_path.exists() and not resume_from:
        restored = restore_checkpoint(cfg, output_dir)
        if restored is not None:
            resume_path = restored
    saved = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path.exists() else None
    initial = None
    initial_cfg = cfg.get("initial_weights", {})
    if saved is None and initial_cfg.get("enabled", False):
        initial_path = hf_hub_download(
            repo_id=initial_cfg["repo_id"], filename=initial_cfg["filename"],
            repo_type="model", revision=initial_cfg.get("revision", "main"),
            cache_dir=ROOT / "data/hf_cache",
        )
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if saved is not None:
        if saved.get("comet_key"):
            cfg["comet"]["resume_experiment_key"] = saved["comet_key"]
            cfg["comet"]["experiment_name"] = saved["experiment_name"]

    old_int = signal.signal(signal.SIGINT, request_pause)
    old_term = signal.signal(signal.SIGTERM, request_pause)
    try:
        with comet_run(cfg) as experiment:
            train(cfg, experiment, saved, initial)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    main()
