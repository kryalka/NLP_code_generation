import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


def resolve_revision(repo_id, revision="main"):
    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha


class HFParquetSource:
    def __init__(self, repo_id, files, revision="main", columns=None):
        self.fs = HfFileSystem()
        self.paths = [f"datasets/{repo_id}@{revision}/{name}" for name in files]
        self.columns = columns
        self._rows = None

    def _parquet(self, path):
        return pq.ParquetFile(self.fs.open(path, "rb"))

    def collect_schema(self):
        schema = self._parquet(self.paths[0]).schema_arrow
        if self.columns is not None:
            schema = pa.schema([schema.field(name) for name in self.columns])
        return pl.from_arrow(pa.Table.from_batches([], schema=schema)).schema

    @property
    def total_rows(self):
        if self._rows is None:
            self._rows = sum(self._parquet(path).metadata.num_rows for path in self.paths)
        return self._rows

    def collect_batches(self, chunk_size, maintain_order=True):
        del maintain_order
        for path in self.paths:
            parquet = self._parquet(path)
            for batch in parquet.iter_batches(batch_size=chunk_size, columns=self.columns):
                yield pl.from_arrow(batch)


class TokenizedHFSource:
    def __init__(self, source, tokenizer):
        self.source = source
        self.tokenizer = tokenizer

    @property
    def total_rows(self):
        return self.source.total_rows

    def collect_batches(self, chunk_size, maintain_order=True):
        for batch in self.source.collect_batches(chunk_size, maintain_order):
            questions = batch["question_clean"].to_list()
            codes = batch["code_clean"].to_list()
            question_ids = [item.ids for item in self.tokenizer.encode_batch(questions)]
            code_ids = [item.ids for item in self.tokenizer.encode_batch(codes)]
            yield pl.DataFrame({"question_ids": question_ids, "code_ids": code_ids})


def read_frame(repo_id, file, revision, columns):
    source = HFParquetSource(repo_id, [file], revision, columns)
    return pl.concat(source.collect_batches(8192), rechunk=True)


def add_token_ids(frame, tokenizer):
    return frame.with_columns(
        pl.Series("question_ids", [item.ids for item in tokenizer.encode_batch(frame["question_clean"].to_list())]),
        pl.Series("code_ids", [item.ids for item in tokenizer.encode_batch(frame["code_clean"].to_list())]),
    )
