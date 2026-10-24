from pathlib import Path

import polars as pl
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKENS = ["<pad>", "</s>"]


def text_batches(path, batch_size):
    data = pl.scan_parquet(path).select("question_clean", "code_clean")

    for batch in data.collect_batches(chunk_size=batch_size, maintain_order=True):
        for column in ("question_clean", "code_clean"):
            texts = batch[column].drop_nulls().to_list()
            if texts:
                yield texts


def train_bpe(path, vocab_size, min_frequency, batch_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_batches(path, batch_size), trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        special_tokens=[("</s>", tokenizer.token_to_id("</s>"))],
    )
    return tokenizer


def main():
    with (ROOT / "configs" / "tokenizers" / "BPE.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    train_path = ROOT / cfg["train_path"]
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for vocab_size in cfg["vocab_sizes"]:
        print(f"Обучаем BPE: {vocab_size}", flush=True)
        tokenizer = train_bpe(
            train_path, vocab_size, cfg["min_frequency"], cfg["batch_size"]
        )
        path = output_dir / f"bpe_{vocab_size}.json"
        tokenizer.save(str(path))
        print(f"Сохранено: {path} (словарь: {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
