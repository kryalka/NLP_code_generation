from pathlib import Path
import random
from time import sleep

import numpy as np
import torch


_PENDING_UPLOADS = set()


def _retry(operation, description, attempts=5):
    for attempt in range(attempts):
        try:
            return True, operation()
        except Exception as error:
            if attempt + 1 == attempts:
                print(f"HF недоступен, отложено: {description}: {error}", flush=True)
                return False, None
            delay = 5 * (3 ** attempt)
            print(f"HF retry через {delay} с: {description}: {error}", flush=True)
            sleep(delay)


def save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_epoch_checkpoint(checkpoint, output_dir, keep=3):
    path = Path(output_dir) / f"epoch_{checkpoint['epoch']:04d}.pt"
    save_checkpoint(checkpoint, path)
    for old in sorted(Path(output_dir).glob("epoch_*.pt"))[:-keep]:
        old.unlink()
    return path


def upload_checkpoint(path, cfg):
    settings = cfg.get("checkpoint_backup", {})
    if not settings.get("enabled", False):
        return True
    from huggingface_hub import batch_bucket_files, list_bucket_tree

    path = Path(path)
    _PENDING_UPLOADS.add(path)
    bucket_id = settings["bucket_id"]
    requested_uploaded = False
    for pending in sorted(_PENDING_UPLOADS):
        if not pending.is_file():
            _PENDING_UPLOADS.discard(pending)
            continue
        remote = f"{settings['run_name']}/{pending.name}"
        success, _ = _retry(
            lambda pending=pending, remote=remote: batch_bucket_files(
                bucket_id, add=[(str(pending), remote)]
            ),
            f"upload {remote}",
        )
        if not success:
            continue
        _PENDING_UPLOADS.discard(pending)
        requested_uploaded |= pending == path
        print(f"HF bucket backup: hf://buckets/{bucket_id}/{remote}", flush=True)
    if path.name.startswith("epoch_") and requested_uploaded:
        prefix = f"{settings['run_name']}/epoch_"
        success, items = _retry(
            lambda: list(list_bucket_tree(
                bucket_id, prefix=settings["run_name"], recursive=True
            )),
            f"list {settings['run_name']}",
        )
        if success:
            epochs = sorted(
                item.path for item in items
                if item.type == "file" and item.path.startswith(prefix) and item.path.endswith(".pt")
            )
            for old in epochs[:-3]:
                _retry(
                    lambda old=old: batch_bucket_files(bucket_id, delete=[old]),
                    f"delete {old}", attempts=3,
                )
    return requested_uploaded


def restore_checkpoint(cfg, output_dir):
    settings = cfg.get("checkpoint_backup", {})
    if not settings.get("enabled", False):
        return None
    from huggingface_hub import download_bucket_files, list_bucket_tree

    bucket_id = settings["bucket_id"]
    remote = f"{settings['run_name']}/resume.pt"
    success, items = _retry(
        lambda: list(list_bucket_tree(
            bucket_id, prefix=settings["run_name"], recursive=True
        )),
        f"restore list {settings['run_name']}",
    )
    if not success:
        raise RuntimeError("Не удалось проверить HF backup; новый запуск запрещён")
    files = {item.path for item in items if item.type == "file"}
    if remote not in files:
        return None
    destination = Path(output_dir) / "resume.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.download")
    temporary.unlink(missing_ok=True)
    success, _ = _retry(
        lambda: download_bucket_files(
            bucket_id, files=[(remote, str(temporary))]
        ),
        f"restore {remote}",
    )
    if not success:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось восстановить {remote}")
    temporary.replace(destination)
    for best_name in ("best_val_loss.pt", "best_pass_at_1.pt"):
        best_remote = f"{settings['run_name']}/{best_name}"
        if best_remote not in files:
            continue
        best_destination = Path(output_dir) / best_name
        best_temporary = best_destination.with_suffix(".pt.download")
        best_temporary.unlink(missing_ok=True)
        success, _ = _retry(
            lambda best_remote=best_remote, best_temporary=best_temporary: download_bucket_files(
                bucket_id, files=[(best_remote, str(best_temporary))]
            ),
            f"restore {best_remote}",
        )
        if success:
            best_temporary.replace(best_destination)
        else:
            best_temporary.unlink(missing_ok=True)
    print(f"Восстановлен HF bucket checkpoint: {remote}", flush=True)
    return destination


def random_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def training_contract(cfg):
    ignored = {"resume_from", "max_seconds", "stop_after_steps", "pass_every_steps"}
    hf_data = dict(cfg.get("hf_data") or {})
    hf_data.pop("local_train_tokens", None)
    return {
        "model": cfg["model"],
        "training": {key: value for key, value in cfg["training"].items() if key not in ignored},
        "generation": cfg["generation"],
        "log_every_steps": cfg["comet"]["log_every_steps"],
        "hf_data": hf_data,
        "initial_weights": cfg.get("initial_weights"),
    }
