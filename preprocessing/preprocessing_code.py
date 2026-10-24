from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import os

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from .code_cleaning import process_chunk

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCH_SIZE = 100000
WORKERS = min(4, os.cpu_count() or 1)

RESULT_SCHEMA = {
    "comment_patch": pl.String, "print_patch": pl.String,
    "is_numeric": pl.Boolean,
}

def clean_wrappers(q):
    """
    Убираем markdown 
    """
    opening = r"(?i)^[ \t\r\n]*```[ \t]*(?:(?:python|py)[ \t]*(?:\r?\n|$)|(?:\r?\n)?)"
    return pl.when(q.str.contains(opening)).then(
        q.str.replace(opening, "").str.replace(r"[ \t]*```[ \t\r\n]*$", "")
    ).otherwise(q.str.replace(r"\n[ \t]*```[ \t\r\n]*$", ""))


def process_train(df, path, progress):
    """
    1) Идём батчами по 100 000 строк
    2) Считаем уже очищенный от markdown код
    3) Нарезаем батч на WORKERS кусков и отдаёт в pool.map(process_chunk, tasks)
    4) Склеиваем результаты обратно в pl.DataFrame со схемой RESULT_SCHEMA
    5) Оставляем программы, для которых подтверждён числовой результат
    """
    source_columns = df.collect_schema().names()
    accepted = 0
    writer = None

    try:
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
            for batch in df.collect_batches(chunk_size=BATCH_SIZE, maintain_order=True):
                # Убираем markdown и отмечаем код с возможными комментариями
                batch = batch.with_columns(clean_wrappers(pl.col("code")).alias("wrapped"))
                codes = batch["wrapped"].to_list()
                flags = (
                    batch["wrapped"]
                    .str.contains("#", literal=True)
                    .fill_null(False)
                    .to_list()
                )

                # Делим батч примерно поровну между процессами
                size = max(1, (batch.height + WORKERS - 1) // WORKERS)
                tasks = []
                for start in range(0, batch.height, size):
                    tasks.append((codes[start:start + size], flags[start:start + size]))

                rows = []
                for chunk in pool.map(process_chunk, tasks):
                    rows.extend(chunk)

                results = pl.DataFrame(rows, schema=RESULT_SCHEMA, orient="row")
                batch = batch.hstack(results)

                # Берём последнюю изменённую версию кода
                batch = batch.with_columns(
                    pl.coalesce(
                        "print_patch", "comment_patch", "wrapped"
                    ).alias("code_clean")
                )
                batch = batch.filter(pl.col("is_numeric"))
                batch = batch.select(*source_columns, "code_clean")
                accepted += batch.height

                if batch.height:
                    table = batch.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression="zstd", use_dictionary=False)
                    writer.write_table(table)

                progress.update(len(codes))
    finally:
        if writer is not None:
            writer.close()

    return accepted


def find_duplicates(df: pl.LazyFrame) -> pl.Series:
    key = pl.struct("question_clean", "code_clean")
    return (
        df.filter(~key.is_first_distinct())
          .select("_row")
          .collect(engine="streaming")
          .get_column("_row")
    )

def write_train(checked, duplicates, path, progress):
    writer = None
    try:
        for batch in checked.collect_batches(chunk_size=BATCH_SIZE, maintain_order=True):
            size = batch.height
            batch = batch.filter(~pl.col("_row").is_in(duplicates.implode()))
            if batch.height:
                table = batch.drop("_row").to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd", use_dictionary=False)
                writer.write_table(table)
            progress.update(size)
    finally:
        if writer is not None:
            writer.close()


def main():
    df_train = pl.scan_parquet(DATA_DIR / "train_clean.parquet").with_row_index("_row")
    n_train = df_train.select(pl.len()).collect().item()

    with TemporaryDirectory(prefix=".code-preprocessing-", dir=DATA_DIR) as tmp_name, \
         tqdm(total=n_train, desc="Очистка кода", unit="строк", dynamic_ncols=True) as progress:

        tmp = Path(tmp_name)
        copyfile(DATA_DIR / "test_clean.parquet", tmp / "test_final.parquet")

        accepted = process_train(df_train, tmp / "checked.parquet", progress)
        progress.refresh()

        checked = pl.scan_parquet(tmp / "checked.parquet")

        progress.set_description_str("Поиск дубликатов", refresh=False)
        progress.reset(total=0)
        duplicates = find_duplicates(checked)

        progress.set_description_str("Сохранение train", refresh=False)
        progress.reset(total=accepted)
        write_train(checked, duplicates, tmp / "train_final.parquet", progress)

        (tmp / "train_final.parquet").replace(DATA_DIR / "train_final.parquet")
        (tmp / "test_final.parquet").replace(DATA_DIR / "test_final.parquet")

        progress.set_description_str("Готово", refresh=False)

if __name__ == "__main__":
    main()
