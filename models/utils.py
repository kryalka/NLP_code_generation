from pathlib import Path
from contextlib import contextmanager
import os

import torch
import yaml
from tokenizers import Tokenizer

from models.baseline import Transformer
from models.code_format import restore_solve_wrapper

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/models/m1_deep_6e6d_d384_100k.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


@contextmanager
def comet_run(cfg, suffix=""):
    settings = cfg.get("comet", {})
    experiment = None
    if settings.get("enabled", False):
        api_key = os.environ.get("COMET_API_KEY")
        if not api_key:
            raise ValueError("УКАЖИ COMET_API_KEY")
        from comet_ml import Experiment

        options = dict(
            api_key=api_key, project_name=settings["project_name"], workspace=settings["workspace"],
            auto_metric_logging=False, auto_param_logging=False, auto_output_logging=False,
            log_code=False, log_graph=False, log_git_metadata=False, log_git_patch=False,
        )
        key = settings.get("resume_experiment_key")
        if key:
            from comet_ml import ExistingExperiment
            experiment = ExistingExperiment(experiment_key=key, **options)
        else:
            experiment = Experiment(**options)
    try:
        if experiment is not None:
            experiment.set_name(settings["experiment_name"] + suffix)
            experiment.log_parameters(cfg)
        yield experiment
    finally:
        if experiment is not None:
            experiment.end()


def load_tokenizer(path, vocab_size):
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer


def load_checkpoint(cfg, device):
    output_dir = ROOT / cfg["output_dir"]
    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    model_cfg = checkpoint["config"]
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = load_tokenizer(output_dir / "tokenizer.json", model_cfg["vocab_size"])
    return model, tokenizer, model_cfg


def prepare_sources(ids, cfg, device):
    limit = cfg["max_source_length"]
    ids = [row if len(row) <= limit else row[:limit - 1] + [1] for row in ids]
    width = max(map(len, ids))
    source = torch.tensor([row + [0] * (width - len(row)) for row in ids], dtype=torch.long)
    return source.to(device)


def prepare_batch(question_ids, code_ids, cfg, device):
    pairs = [
        (question, code) for question, code in zip(question_ids, code_ids)
        if len(code) <= cfg["max_target_length"]
    ]
    if not pairs:
        return None
    question_ids, code_ids = map(list, zip(*pairs))
    source = prepare_sources(question_ids, cfg, device)
    width = max(map(len, code_ids))
    target = torch.tensor(
        [row + [0] * (width - len(row)) for row in code_ids],
        dtype=torch.long, device=device
    )
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]
    labels = target.masked_fill(target == 0, -100)
    return source, decoder_input, labels


def generate_ids(model, tokenizer, question_ids, cfg, max_new_tokens, device):
    source = prepare_sources(question_ids, cfg, device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        generated = model.generate(source, max_new_tokens=max_new_tokens)
    codes = []
    for ids in generated.tolist():
        if 1 in ids:
            ids = ids[:ids.index(1)]
        code = tokenizer.decode(ids, skip_special_tokens=True)
        if cfg.get("body_only_targets", False):
            code = restore_solve_wrapper(code)
        codes.append(code)
    return codes


def generate_codes(model, tokenizer, questions, cfg, max_new_tokens, device):
    question_ids = [item.ids for item in tokenizer.encode_batch(questions)]
    return generate_ids(model, tokenizer, question_ids, cfg, max_new_tokens, device)
