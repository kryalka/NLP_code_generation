from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

from models.code_format import restore_solve_wrapper


def restore_token_wrappers(source, destination, tokenizer_path, batch_size=8192):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(".partial.parquet")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    parquet = pq.ParquetFile(source)
    code_index = parquet.schema_arrow.get_field_index("code_ids")
    writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
    try:
        with tqdm(total=parquet.metadata.num_rows, desc="Восстановление обёрток", unit="строк") as progress:
            for batch in parquet.iter_batches(batch_size=batch_size):
                code_ids = batch.column(code_index).to_pylist()
                bodies = tokenizer.decode_batch(code_ids)
                full_ids = [item.ids for item in tokenizer.encode_batch(
                    [restore_solve_wrapper(body) for body in bodies]
                )]
                batch = batch.set_column(
                    code_index, "code_ids",
                    pa.array(full_ids, type=parquet.schema_arrow.field(code_index).type),
                )
                writer.write_batch(batch)
                progress.update(batch.num_rows)
    finally:
        writer.close()
    temporary.replace(destination)
    return destination
