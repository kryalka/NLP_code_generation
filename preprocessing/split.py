from contextlib import ExitStack
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_DIR = DATA_DIR / "splits"

SEED = 42
VAL_MOD = 5000
SAMPLE_SEED = 43
SAMPLE_MOD = 1000000
SMALL_TARGET_ROWS = 500000
LARGE_TARGET_ROWS = 2000000
BATCH_SIZE = 50000
SHUFFLE_BUCKETS = 64
TARGET_REPO = "kryalka/my-training-data"


def write_shuffled(df, path):
    rng = np.random.default_rng(SEED)
    with TemporaryDirectory(prefix=".shuffle-", dir=path.parent) as directory:
        temporary = Path(directory)
        source = temporary / "source.parquet"
        df.sink_parquet(source, row_group_size=BATCH_SIZE, engine="streaming")
        buckets = [temporary / f"bucket_{i}.parquet" for i in range(SHUFFLE_BUCKETS)]

        with pq.ParquetFile(source) as reader:
            schema = reader.schema_arrow
            total = reader.metadata.num_rows
            with ExitStack() as stack, tqdm(
                total=total, desc=f"{path.stem}: распределение", unit="строк",
                dynamic_ncols=True, leave=False,
            ) as progress:
                writers = [stack.enter_context(pq.ParquetWriter(p, schema, compression="zstd")) for p in buckets]
                for batch in reader.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
                    table = pa.Table.from_batches([batch])
                    indices = rng.integers(SHUFFLE_BUCKETS, size=table.num_rows)
                    for i, writer in enumerate(writers):
                        part = table.filter(pa.array(indices == i))
                        if part.num_rows:
                            writer.write_table(part)
                    progress.update(table.num_rows)
                    del table, part, batch
        source.unlink()

        result = temporary / "result.parquet"
        with pq.ParquetWriter(result, schema, compression="zstd") as writer, tqdm(
            total=total, desc=f"{path.stem}: перемешивание", unit="строк",
            dynamic_ncols=True, leave=False,
        ) as progress:
            for i in rng.permutation(SHUFFLE_BUCKETS):
                with pq.ParquetFile(buckets[i]) as reader:
                    table = reader.read(use_threads=False)
                table = table.take(pa.array(rng.permutation(table.num_rows)))
                writer.write_table(table, row_group_size=BATCH_SIZE)
                progress.update(table.num_rows)
                del table
                buckets[i].unlink()
        result.replace(path)


def main():
    df_train = pl.scan_parquet(DATA_DIR / "train_final.parquet")

    question_key = (
        pl.col("question_clean")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_lowercase()
    )
    df_split = df_train.with_columns(
        (question_key.hash(seed=SEED) % VAL_MOD).alias("_split")
    )
    df_val = df_split.filter(pl.col("_split") == 0).drop("_split")
    df_train = df_split.filter(
        pl.col("_split").is_null() | (pl.col("_split") != 0)
    ).drop("_split")

    with tqdm(total=6, desc="Подсчёт train", unit="этап", dynamic_ncols=True) as progress:
        full_train_rows = df_train.select(pl.len()).collect(engine="streaming").item()
        small_thr = round(SMALL_TARGET_ROWS / full_train_rows * SAMPLE_MOD)
        large_thr = round(LARGE_TARGET_ROWS / full_train_rows * SAMPLE_MOD)
        df_ranked = df_train.with_columns(
            (question_key.hash(seed=SAMPLE_SEED) % SAMPLE_MOD).alias("_bucket")
        )
        df_small = df_ranked.filter(pl.col("_bucket") < small_thr).drop("_bucket")
        df_large = df_ranked.filter(pl.col("_bucket") < large_thr).drop("_bucket")
        progress.update(1)

        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        progress.set_description_str("Сохранение val")
        df_val.sink_parquet(SPLITS_DIR / "val.parquet", engine="streaming")
        progress.update(1)

        for name, frame in (("train_small", df_small), ("train_large", df_large), ("train_full", df_train)):
            progress.set_description_str(f"Подготовка и перемешивание {name}")
            write_shuffled(frame, SPLITS_DIR / f"{name}.parquet")
            progress.update(1)

        progress.set_description_str("Копирование test")
        copyfile(DATA_DIR / "test_final.parquet", SPLITS_DIR / "test.parquet")
        progress.update(1)
        progress.set_description_str("Загрузка на HF")

    files = ["train_small.parquet", "train_large.parquet", "train_full.parquet",
             "val.parquet", "test.parquet"]
    api = HfApi()
    commit = api.create_commit(
        repo_id=TARGET_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=SPLITS_DIR / name)
                    for name in files],
        commit_message="Create deterministic cleaned train/val/test splits",
    )
    print(f"HF splits: {TARGET_REPO}@{commit.oid}")
    for name in files:
        (SPLITS_DIR / name).unlink()
    for name in ("train_clean.parquet", "test_clean.parquet",
                 "train_final.parquet", "test_final.parquet"):
        (DATA_DIR / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
