from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import polars as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.hf_data import add_token_ids, read_frame, resolve_revision
from models.code_format import restore_solve_wrapper
from models.tokenization import tokenize_file
from models.utils import ROOT, comet_run, generate_ids, load_checkpoint, load_config, prepare_batch
from val.execute import run_code


def reference_answers(val, body_only=False):
    codes = val["code_clean"]
    if body_only:
        codes = [restore_solve_wrapper(code) for code in codes]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(tqdm(pool.map(run_code, codes), total=val.height, desc="Эталоны val"))


def evaluate(model, tokenizer, val, answers, cfg, device, compute_pass=True):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    was_training = model.training
    model.eval()
    model_cfg = cfg["model"]

    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for batch in tqdm(val.iter_slices(n_rows=cfg["training"]["batch_size"]), desc="Val loss"):
            prepared = prepare_batch(
                batch["question_ids"].to_list(), batch["code_ids"].to_list(), model_cfg, device
            )
            if prepared is None:
                continue

            source, decoder_input, labels = prepared
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(source, decoder_input, labels)
                loss = F.cross_entropy(logits.float(), labels[labels.ne(-100)], reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.ne(-100).sum().item()
            del logits, loss
            
    metrics = {
        "val_loss": total_loss / total_tokens,
        "rows": val.height,
        "loss_seconds": perf_counter() - started,
    }
    if compute_pass:
        correct = 0
        wrong_answer = 0
        execution_failed = 0
        reference_failed = 0
        generation_seconds = 0.0
        pending = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            batch_size = cfg["generation"]["batch_size"]
            for start in tqdm(range(0, val.height, batch_size), desc="Val pass@1"):
                question_ids = val.slice(start, batch_size)["question_ids"].to_list()
                generation_started = perf_counter()
                codes = generate_ids(
                    model, tokenizer, question_ids, model_cfg,
                    cfg["generation"]["max_new_tokens"], device,
                )
                generation_seconds += perf_counter() - generation_started
                for code, expected in zip(codes, answers[start:start + batch_size]):
                    pending.append((pool.submit(run_code, code), expected))

            execution_started = perf_counter()
            for future, expected in pending:
                predicted = future.result()
                if expected is None:
                    reference_failed += 1
                elif predicted is None:
                    execution_failed += 1
                elif abs(predicted - expected) <= 1e-6 + 1e-6 * abs(expected):
                    correct += 1
                else:
                    wrong_answer += 1
            execution_wait_seconds = perf_counter() - execution_started
        metrics.update({
            "pass_at_1": correct / val.height,
            "correct": correct,
            "incorrect": wrong_answer + execution_failed,
            "wrong_answer": wrong_answer,
            "execution_failed": execution_failed,
            "reference_failed": reference_failed,
            "reference_ok": sum(answer is not None for answer in answers),
            "generation_seconds": generation_seconds,
            "execution_wait_seconds": execution_wait_seconds,
        })
    model.train(was_training)
    metrics["seconds"] = perf_counter() - started
    return metrics


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_cfg = load_checkpoint(cfg, device)
    cfg["model"] = model_cfg
    with comet_run(cfg, suffix="-val") as experiment:
        hf = cfg.get("hf_data", {})
        tokenizer_path = ROOT / cfg["output_dir"] / "tokenizer.json"
        if hf.get("enabled", False):
            revision = resolve_revision(hf["repo_id"], hf.get("revision", "main"))
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
        else:
            val = pl.read_parquet(ROOT / cfg["val_path"], columns=["question_clean", "code_clean"])
            val_tokens = tokenize_file(ROOT / cfg["val_path"], tokenizer_path, cfg)
            val = val.hstack(pl.read_parquet(val_tokens))
        answers = reference_answers(val, model_cfg.get("body_only_targets", False))
        metrics = evaluate(model, tokenizer, val, answers, cfg, device)
        print(metrics)
        if experiment is not None:
            values = dict(metrics)
            values["loss"] = values.pop("val_loss")
            experiment.log_metrics(values, prefix="val", step=0)


if __name__ == "__main__":
    main()
