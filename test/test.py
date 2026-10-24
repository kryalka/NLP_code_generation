import argparse
import json

import polars as pl
import torch
from tqdm import tqdm

from models.hf_data import read_frame, resolve_revision
from models.utils import ROOT, generate_codes, load_checkpoint, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/m1_deep_6e6d_d384_100k.yaml")
    cfg = load_config(parser.parse_args().config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_cfg = load_checkpoint(cfg, device)

    hf = cfg.get("hf_data", {})
    if hf.get("enabled", False):
        revision = resolve_revision(hf["repo_id"], hf.get("revision", "main"))
        questions = read_frame(hf["repo_id"], hf["test_file"], revision, ["question_clean"])
    else:
        questions = pl.read_parquet(ROOT / cfg["test_path"], columns=["question_clean"])
    questions = questions["question_clean"].to_list()
    
    batch_size = cfg["generation"]["batch_size"]
    output_path = ROOT / cfg["output_dir"] / "test_predictions.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for start in tqdm(range(0, len(questions), batch_size), desc="Test"):
            codes = generate_codes(
                model, tokenizer, questions[start:start + batch_size], model_cfg,
                cfg["generation"]["max_new_tokens"], device,
            )
            for code in codes:
                file.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")
    print(f"Сохранено: {output_path}; строк: {len(questions)}")


if __name__ == "__main__":
    main()
