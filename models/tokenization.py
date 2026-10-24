from hashlib import file_digest, sha256
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def copy_cache(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".copy-tokens-", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        copyfile(source, temporary)
        temporary.replace(destination)


def tokenize_file(path, tokenizer_path, cfg):
    settings = cfg.get("tokenization", {})
    cache_dir = ROOT / settings.get("cache_dir", "data/token_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = f"v1:{file_hash(path)}:{file_hash(tokenizer_path)}:{tokenizers.__version__}"
    key = sha256(signature.encode()).hexdigest()
    cache_path = cache_dir / f"{path.stem}_{key}.parquet"
    backup_dir = settings.get("backup_dir")
    backup_path = ROOT / backup_dir / cache_path.name if backup_dir else None

    if not cache_path.exists():
        for directory in settings.get("input_cache_dirs", []):
            candidate = Path(directory) / cache_path.name
            if candidate.is_file():
                copy_cache(candidate, cache_path)
                break

    if not cache_path.exists() and backup_path is not None and backup_path.exists():
        copy_cache(backup_path, cache_path)

    if not cache_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        source = pl.scan_parquet(path).select("question_clean", "code_clean")
        total = source.select(pl.len()).collect().item()
        schema = pa.schema([
            ("question_ids", pa.list_(pa.uint32())),
            ("code_ids", pa.list_(pa.uint32())),
        ])
        with TemporaryDirectory(prefix=".tokenize-", dir=cache_dir) as directory:
            temporary = Path(directory) / "tokens.parquet"
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                with tqdm(total=total, desc=f"Токенизация {path.stem}", unit="строк") as progress:
                    for batch in source.collect_batches(chunk_size=8192, maintain_order=True):
                        columns = {}
                        for name, target in (("question_clean", "question_ids"), ("code_clean", "code_ids")):
                            texts = batch[name].to_list()
                            encoded = iter(tokenizer.encode_batch([text for text in texts if text is not None]))
                            columns[target] = [None if text is None else next(encoded).ids for text in texts]
                        writer.write_table(pa.Table.from_pydict(columns, schema=schema))
                        progress.update(batch.height)
            temporary.replace(cache_path)

    if backup_path is not None and not backup_path.exists():
        copy_cache(cache_path, backup_path)
    print("Кеш токенов:", cache_path)
    return cache_path
