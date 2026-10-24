from concurrent.futures import ThreadPoolExecutor
import random

import polars as pl
import torch

from models.utils import prepare_batch


def batch_spans(chunk, settings, model_cfg):
    token_budget = settings.get("max_tokens_per_batch")
    if token_budget is None:
        size = settings["batch_size"]
        return [(start, min(size, chunk.height - start)) for start in range(0, chunk.height, size)]

    max_examples = settings.get("max_examples_per_batch", 256)
    source_lengths = chunk["question_ids"].list.len().clip(upper_bound=model_cfg["max_source_length"]).to_list()
    target_lengths = chunk["code_ids"].list.len().clip(upper_bound=model_cfg["max_target_length"]).to_list()
    spans = []
    start = count = source_max = target_max = 0
    for source_length, target_length in zip(source_lengths, target_lengths):
        next_count = count + 1
        next_source_max = max(source_max, source_length)
        next_target_max = max(target_max, target_length)
        padded_tokens = next_count * (next_source_max + next_target_max)
        if count and (padded_tokens > token_budget or count >= max_examples):
            spans.append((start, count))
            start += count
            count = source_max = target_max = 0
            next_count, next_source_max, next_target_max = 1, source_length, target_length
        count = next_count
        source_max = next_source_max
        target_max = next_target_max
    if count:
        spans.append((start, count))
    return spans


def batches_from_cursor(source, settings, model_cfg, cursor, total_rows, pin_memory):
    epoch = cursor["epoch"]
    offset = 0
    chunks = source.collect_batches(chunk_size=settings["chunk_size"], maintain_order=True)
    for index, chunk in enumerate(chunks):
        offset += chunk.height
        if index < cursor["chunk"]:
            continue
        seed = settings["seed"] + epoch + index
        chunk = chunk.sample(fraction=1.0, shuffle=True, seed=seed).sort(
            pl.col("code_ids").list.len(), pl.col("question_ids").list.len(), maintain_order=True)
        spans = batch_spans(chunk, settings, model_cfg)
        random.Random(seed).shuffle(spans)
        for batch_index, (start, size) in enumerate(spans):
            if index == cursor["chunk"] and batch_index < cursor["batch"]:
                continue
            batch = chunk.slice(start, size)
            prepared = prepare_batch(
                batch["question_ids"].to_list(), batch["code_ids"].to_list(),
                model_cfg, torch.device("cpu"),
            )
            tokens = 0
            if prepared is not None:
                tokens = prepared[2].ne(-100).sum().item()
                if pin_memory:
                    prepared = tuple(tensor.pin_memory() for tensor in prepared)
            padded_tokens = 0
            sequence_tokens = 0
            if prepared is not None:
                source_tensor, _, labels = prepared
                sequence_tokens = source_tensor.ne(0).sum().item() + tokens
                padded_tokens = source_tensor.numel() + labels.numel()
            last_batch = batch_index + 1 == len(spans)
            epoch_end = last_batch and offset == total_rows
            if epoch_end:
                next_cursor = {"epoch": epoch + 1, "chunk": 0, "batch": 0}
            elif last_batch:
                next_cursor = {"epoch": epoch, "chunk": index + 1, "batch": 0}
            else:
                next_cursor = {"epoch": epoch, "chunk": index, "batch": batch_index + 1}
            yield prepared, batch.height, tokens, padded_tokens, sequence_tokens, next_cursor, epoch_end


def prefetch_from_cursor(source, settings, model_cfg, cursor, total_rows, pin_memory=False):
    batches = batches_from_cursor(source, settings, model_cfg, cursor, total_rows, pin_memory)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(next, batches, None)
            while True:
                batch = future.result()
                if batch is None:
                    break
                future = pool.submit(next, batches, None)
                yield batch
    finally:
        batches.close()
