import argparse
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
from tokenizers import Tokenizer
from tqdm import tqdm

from models.code_format import strip_solve_wrapper
from models.utils import ROOT


def require_write_token(api, repo_id):
    identity = api.whoami()
    access = identity.get("auth", {}).get("accessToken", {})
    role = access.get("role")
    if role in {"write", "admin"}:
        return
    owner = repo_id.split("/", 1)[0]
    if role == "fineGrained":
        scopes = access.get("fineGrained", {}).get("scoped", [])
        for scope in scopes:
            entity = scope.get("entity", {})
            permissions = set(scope.get("permissions", []))
            if entity.get("name") == owner and permissions & {
                "repo.write", "repo.content.write", "repo.contribute.write",
            }:
                return
    raise PermissionError(
        "Текущий HF_TOKEN не имеет права записи. Выполни `hf auth login` "
        "с write-токеном до запуска обработки."
    )


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def transform_file(source, destination, tokenizer=None, token_path=None, batch_size=8192):
    with Path(source).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        code_index = parquet.schema_arrow.get_field_index("code_clean")
        question_index = parquet.schema_arrow.get_field_index("question_clean")
        writer = pq.ParquetWriter(destination, parquet.schema_arrow, compression="zstd")
        token_writer = None
        if token_path is not None:
            token_schema = pa.schema([
                ("question_ids", pa.list_(pa.uint32())),
                ("code_ids", pa.list_(pa.uint32())),
            ])
            token_writer = pq.ParquetWriter(token_path, token_schema, compression="zstd")
        changed = 0
        try:
            with tqdm(total=parquet.metadata.num_rows, desc=Path(destination).stem, unit="строк") as progress:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    codes = batch.column(code_index).to_pylist()
                    bodies = [strip_solve_wrapper(code) for code in codes]
                    changed += sum(code != body for code, body in zip(codes, bodies))
                    batch = batch.set_column(
                        code_index, "code_clean",
                        pa.array(bodies, type=parquet.schema_arrow.field(code_index).type),
                    )
                    writer.write_batch(batch)
                    if token_writer is not None:
                        questions = batch.column(question_index).to_pylist()
                        question_ids = [item.ids for item in tokenizer.encode_batch(questions)]
                        code_ids = [item.ids for item in tokenizer.encode_batch(bodies)]
                        token_writer.write_table(pa.Table.from_pydict(
                            {"question_ids": question_ids, "code_ids": code_ids},
                            schema=token_schema,
                        ))
                    progress.update(batch.num_rows)
        finally:
            writer.close()
            if token_writer is not None:
                token_writer.close()
    return parquet.metadata.num_rows, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="kryalka/my-training-data")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train-file", default="train_full.parquet")
    parser.add_argument("--val-file", default="val.parquet")
    parser.add_argument("--tokens-file", default="tokens/train_full_bpe_16384.parquet")
    parser.add_argument("--tokenizer", default="artifacts/tokenizers/bpe_16384.json")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise PermissionError("Сначала выполни `hf auth login` с write-токеном")
    api = HfApi(token=token)
    require_write_token(api, args.repo_id)
    revision = api.repo_info(args.repo_id, repo_type="dataset", revision=args.revision).sha
    work_dir = ROOT / "data"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "hf_cache"
    staging = work_dir / "body_only_hf_staging"
    signature = {
        "format": 1, "repo_id": args.repo_id, "source_revision": revision,
        "train_file": args.train_file, "val_file": args.val_file,
        "tokens_file": args.tokens_file, "tokenizer": args.tokenizer,
    }
    state_path = staging / "state.json"
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("signature") != signature:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        state = {"signature": signature}
        save_state(state_path, state)

    tokenizer = Tokenizer.from_file(str(ROOT / args.tokenizer))
    train_out = staging / "train.parquet"
    val_out = staging / "val.parquet"
    tokens_out = staging / "tokens.parquet"
    if not (state.get("train_complete") and train_out.is_file() and tokens_out.is_file()):
        print("Скачивание исходного train в возобновляемый HF-кэш")
        train_source = hf_hub_download(
            repo_id=args.repo_id, repo_type="dataset", filename=args.train_file,
            revision=revision, cache_dir=cache_dir, token=token,
        )
        train_partial = staging / "train.partial.parquet"
        tokens_partial = staging / "tokens.partial.parquet"
        train_rows, train_changed = transform_file(
            train_source, train_partial,
            tokenizer, tokens_partial, args.batch_size,
        )
        train_partial.replace(train_out)
        tokens_partial.replace(tokens_out)
        state.update(train_complete=True, train_rows=train_rows, train_changed=train_changed)
        save_state(state_path, state)
    else:
        train_rows, train_changed = state["train_rows"], state["train_changed"]
        print("Train уже подготовлен, повторная обработка не нужна")

    if not (state.get("val_complete") and val_out.is_file()):
        val_source = hf_hub_download(
            repo_id=args.repo_id, repo_type="dataset", filename=args.val_file,
            revision=revision, cache_dir=cache_dir, token=token,
        )
        val_partial = staging / "val.partial.parquet"
        val_rows, val_changed = transform_file(
            val_source, val_partial,
            batch_size=args.batch_size,
        )
        val_partial.replace(val_out)
        state.update(val_complete=True, val_rows=val_rows, val_changed=val_changed)
        save_state(state_path, state)
    else:
        val_rows, val_changed = state["val_rows"], state["val_changed"]
        print("Val уже подготовлен, повторная обработка не нужна")

    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=args.train_file, path_or_fileobj=train_out),
            CommitOperationAdd(path_in_repo=args.val_file, path_or_fileobj=val_out),
            CommitOperationAdd(path_in_repo=args.tokens_file, path_or_fileobj=tokens_out),
        ],
        commit_message="Store function bodies and rebuild tokenized train",
    )
    print({
        "source_revision": revision,
        "new_revision": commit.oid,
        "train_rows": train_rows,
        "train_changed": train_changed,
        "val_rows": val_rows,
        "val_changed": val_changed,
        "tokens_file": args.tokens_file,
    })
    shutil.rmtree(staging)


if __name__ == "__main__":
    main()
