import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download

from models.utils import ROOT, load_config
from preprocessing.restore_token_wrappers import restore_token_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/m1_deep_6e6d_d384_100k.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    hf = cfg["hf_data"]

    body_tokens = hf_hub_download(
        repo_id=hf["repo_id"],
        filename=hf["body_train_tokens_file"],
        repo_type="dataset",
        revision="main",
        cache_dir=ROOT / "data/hf_cache",
    )
    with TemporaryDirectory(prefix="wrapper-tokens-", dir=ROOT / "data") as directory:
        tokens = Path(directory) / "train_full_wrapper_bpe_16384.parquet"
        restore_token_wrappers(
            body_tokens, tokens, ROOT / cfg["tokenizer_path"]
        )
        commit = HfApi().upload_file(
            path_or_fileobj=str(tokens),
            path_in_repo=hf["train_tokens_file"],
            repo_id=hf["repo_id"],
            repo_type="dataset",
        )
    print(f"Tokenized train uploaded: {hf['repo_id']}@{commit.oid}")


if __name__ == "__main__":
    main()
