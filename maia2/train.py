import hashlib
import json
import os
from multiprocessing import cpu_count, get_context
from pathlib import Path
from queue import Empty
import random
import tempfile
import time
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn

from . import __version__
from .main import MAIA2Model, preprocess_thread, read_monthly_data_path, train_chunks
from .utils import count_parameters, create_elo_dict, get_all_possible_moves
from .utils import decompression_provenance_path, decompress_zst
from .utils import readable_num, readable_time, read_decompression_provenance
from .utils import read_or_create_chunks, seed_everything, sha256_file


_WORKER_ERROR = "maia2_preprocessing_error"
_WORKER_POLL_SECONDS = 1.0
_RUN_MANIFEST_NAME = "run_manifest.json"
_MISSING = "__MAIA2_CONFIG_VALUE_MISSING__"
_UNSET = object()

# Bump this identifier whenever game_filter or process_per_game changes which
# games or positions are admitted. A run must never resume across incompatible
# filtering semantics merely because its numeric configuration is unchanged.
_TRAINING_FILTER_POLICY = "rated-rapid-bot-title-clock-v1"

_CRITICAL_CONFIG_DEFAULTS = {
    # The legacy released configuration predates this explicit option. Keep
    # its historical behavior stable while making the value part of manifests.
    "skip_months": ["2019-12"],
}

_CRITICAL_CONFIG_GROUPS = {
    "architecture": (
        "input_channels",
        "dim_cnn",
        "num_blocks_cnn",
        "vit_length",
        "dim_vit",
        "num_blocks_vit",
        "elo_dim",
    ),
    "filters": (
        "first_n_moves",
        "last_n_moves",
        "max_ply",
        "clock_threshold",
        "max_games_per_elo_range",
    ),
    "losses": (
        "side_info",
        "side_info_coefficient",
        "value",
        "value_coefficient",
    ),
    "optimizer": ("lr", "wd", "batch_size"),
    "reproducibility": ("seed",),
    "data_pipeline": (
        "start_year",
        "start_month",
        "end_year",
        "end_month",
        "skip_months",
        "chunk_size",
        "num_cpu_left",
    ),
}


def resolve_device(device="auto"):
    if isinstance(device, torch.device):
        requested_device = device
    else:
        device = "auto" if device is None else str(device).lower()
        if device == "gpu":
            device = "cuda"

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        requested_device = torch.device(device)

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    if requested_device.type == "cuda" and requested_device.index is not None:
        device_count = torch.cuda.device_count()
        if not 0 <= requested_device.index < device_count:
            raise RuntimeError(
                f"CUDA device index {requested_device.index} was requested, but "
                f"only {device_count} CUDA device(s) are visible."
            )
    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available.")
    if requested_device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported training device: {requested_device.type}. "
            "Choose from 'auto', 'cpu', 'cuda', or 'mps'."
        )

    return requested_device


def load_model_state_dict(model, state_dict):
    target_uses_data_parallel = isinstance(model, nn.DataParallel)
    checkpoint_uses_data_parallel = any(key.startswith("module.") for key in state_dict)

    if checkpoint_uses_data_parallel and not target_uses_data_parallel:
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    elif target_uses_data_parallel and not checkpoint_uses_data_parallel:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)


def _should_use_data_parallel(device):
    """Use legacy DataParallel only for an unindexed multi-GPU request."""

    return (
        device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1
    )


def get_num_processes(num_cpu_left):
    return max(1, cpu_count() - num_cpu_left)


def _preprocess_worker(result_queue, cfg, pgn_path, pgn_chunks, elo_dict):
    """Run preprocessing and return child-process failures to the parent."""

    try:
        preprocess_thread(
            result_queue,
            cfg,
            pgn_path,
            pgn_chunks,
            elo_dict,
        )
    except BaseException:
        result_queue.put(
            {
                "type": _WORKER_ERROR,
                "traceback": traceback.format_exc(),
            }
        )


def _start_preprocessing_worker(
    context, result_queue, cfg, pgn_path, pgn_chunks, elo_dict
):
    worker = context.Process(
        target=_preprocess_worker,
        args=(result_queue, cfg, pgn_path, pgn_chunks, elo_dict),
    )
    worker.start()
    return worker


def _wait_for_preprocessing_result(result_queue, worker):
    """Wait without polling Queue.empty(), detecting silent worker exits."""

    while True:
        try:
            result = result_queue.get(timeout=_WORKER_POLL_SECONDS)
            break
        except Empty:
            if worker.is_alive():
                continue
            worker.join()
            try:
                # A multiprocessing Queue can briefly lag behind process exit
                # while its feeder thread flushes the final object.
                result = result_queue.get(timeout=_WORKER_POLL_SECONDS)
                break
            except Empty:
                raise RuntimeError(
                    "The preprocessing worker exited without returning data "
                    f"(exit code {worker.exitcode})."
                ) from None

    worker.join()
    if isinstance(result, dict) and result.get("type") == _WORKER_ERROR:
        raise RuntimeError("The preprocessing worker failed:\n" + result["traceback"])
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise RuntimeError(
            f"The preprocessing worker returned an invalid result: {result!r}"
        )
    return result


def _iter_preprocessed_batches(cfg, pgn_path, pgn_chunks_sublists, elo_dict):
    """Yield preprocessed batches while overlapping the next preprocessing job."""

    if not pgn_chunks_sublists:
        return

    start_method = getattr(cfg, "multiprocessing_start_method", "spawn")
    context = get_context(start_method)
    result_queue = context.Queue(maxsize=cfg.queue_length)
    worker = None
    try:
        worker = _start_preprocessing_worker(
            context,
            result_queue,
            cfg,
            pgn_path,
            pgn_chunks_sublists[0],
            elo_dict,
        )
        for index in range(len(pgn_chunks_sublists)):
            result = _wait_for_preprocessing_result(result_queue, worker)
            worker = None

            if index + 1 < len(pgn_chunks_sublists):
                worker = _start_preprocessing_worker(
                    context,
                    result_queue,
                    cfg,
                    pgn_path,
                    pgn_chunks_sublists[index + 1],
                    elo_dict,
                )

            yield result
    finally:
        if worker is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        result_queue.close()
        result_queue.join_thread()


def _checkpoint_name(epoch, pgn_path):
    month = Path(pgn_path).name.removeprefix("lichess_db_standard_rated_")
    return f"epoch_{epoch}_{month}.pt"


def _optimizer_step_count(optimizer):
    """Return Adam-style optimizer steps without assuming a tensor device."""

    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            return int(step.item() if torch.is_tensor(step) else step)
    return 0


def _require_trained_month(pgn_path, positions, games, optimizer_steps):
    if positions > 0 and games > 0 and optimizer_steps > 0:
        return
    raise RuntimeError(
        f"Training accepted no usable data from {pgn_path}: "
        f"positions={positions}, games={games}, "
        f"optimizer_steps={optimizer_steps}. The decompressed PGN is being "
        "kept and no checkpoint will be written. Check the game filters and "
        "source archive before continuing."
    )


def _save_checkpoint_atomic(checkpoint, destination, *, overwrite=False):
    """Write a checkpoint atomically without silently replacing an artifact."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint {destination}. Set "
            "overwrite_checkpoints=true only when replacing it is intentional."
        )
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.save(checkpoint, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            # A hard link is an atomic no-clobber install because both paths are
            # in the destination directory. It closes the exists()/replace()
            # race that could otherwise overwrite a checkpoint from another run.
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                raise FileExistsError(
                    f"Refusing to overwrite existing checkpoint {destination}. "
                    "Set overwrite_checkpoints=true only when replacing it is "
                    "intentional."
                ) from None
            temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _config_value(config, key):
    default = _CRITICAL_CONFIG_DEFAULTS.get(key, _MISSING)
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _critical_config(config):
    """Return the canonical settings that define checkpoint compatibility."""

    critical = {
        group: {key: _json_safe(_config_value(config, key)) for key in keys}
        for group, keys in _CRITICAL_CONFIG_GROUPS.items()
    }
    critical["filters"]["policy"] = _TRAINING_FILTER_POLICY
    critical["optimizer"]["type"] = "AdamW"
    return critical


def _run_manifest(cfg):
    critical = _critical_config(cfg)
    canonical = json.dumps(critical, sort_keys=True, separators=(",", ":"))
    return {
        "format_version": 1,
        "critical_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "critical_config": critical,
    }


def _write_json_atomic(value, destination, *, no_clobber=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(value, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    try:
        if no_clobber:
            os.link(temporary_path, destination)
            temporary_path.unlink()
        else:
            os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _ensure_run_manifest(save_root, cfg, *, allow_legacy_resume=False):
    """Create or validate the critical-config manifest for a checkpoint run."""

    save_root = Path(save_root)
    manifest_path = save_root / _RUN_MANIFEST_NAME
    expected = _run_manifest(cfg)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read run manifest {manifest_path}: {error}"
            ) from error
        if existing != expected:
            raise RuntimeError(
                f"Critical training configuration does not match {manifest_path}. "
                "Use a different save_root; mixing incompatible checkpoints in "
                "one run directory is not supported."
            )
        return manifest_path

    existing_checkpoints = list(save_root.glob("epoch_*.pt"))
    if existing_checkpoints and not allow_legacy_resume:
        raise RuntimeError(
            f"Checkpoint directory {save_root} has checkpoint files but no run "
            "manifest. Refusing a fresh run because its configuration cannot be "
            "compared. Resume the legacy checkpoint or use a new save_root."
        )
    if existing_checkpoints:
        warnings.warn(
            f"Creating a run manifest for a validated legacy resume in {save_root}; existing "
            "checkpoints predate manifest-based collision protection.",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        _write_json_atomic(expected, manifest_path, no_clobber=True)
    except FileExistsError:
        # Another process won the atomic install. Validate what it wrote rather
        # than replacing an incompatible manifest.
        return _ensure_run_manifest(
            save_root,
            cfg,
            allow_legacy_resume=allow_legacy_resume,
        )
    return manifest_path


def _validate_checkpoint_destinations(
    save_root, training_schedule, *, overwrite_checkpoints=False
):
    """Fail before training if the schedule would overwrite a checkpoint."""

    if overwrite_checkpoints:
        return
    collisions = [
        Path(save_root) / _checkpoint_name(epoch + 1, pgn_path)
        for epoch, pgn_paths in training_schedule
        for pgn_path in pgn_paths
        if (Path(save_root) / _checkpoint_name(epoch + 1, pgn_path)).exists()
    ]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions[:3])
        if len(collisions) > 3:
            rendered += f", and {len(collisions) - 3} more"
        raise FileExistsError(
            f"The training schedule would overwrite existing checkpoint(s): "
            f"{rendered}. Set overwrite_checkpoints=true only when replacement "
            "is intentional."
        )


def _capture_rng_state(device):
    """Capture RNG state needed for a checkpoint-equivalent resume."""

    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (
        device.type == "mps"
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state):
    """Restore RNG state saved by :func:`_capture_rng_state`."""

    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_cuda_states = [item.cpu() for item in state["torch_cuda"]]
        visible_cuda_devices = torch.cuda.device_count()
        if len(saved_cuda_states) != visible_cuda_devices:
            warnings.warn(
                "The checkpoint saved CUDA RNG state for "
                f"{len(saved_cuda_states)} device(s), but "
                f"{visible_cuda_devices} are visible. Restoring the shared "
                "device prefix; exact multi-GPU RNG replay is not possible.",
                RuntimeWarning,
                stacklevel=2,
            )
        for index, cuda_state in enumerate(saved_cuda_states[:visible_cuda_devices]):
            torch.cuda.set_rng_state(cuda_state, device=index)
    if (
        "torch_mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["torch_mps"].cpu())


def _config_snapshot(cfg):
    """Return a weights-only-safe snapshot of a training configuration."""

    return _json_safe(vars(cfg))


def _training_metadata(
    cfg,
    pgn_path,
    epoch,
    optimizer_steps,
    source_sha256=None,
    decompressed_sha256=None,
    source_provenance=None,
):
    archive_path = Path(pgn_path + ".zst")
    pgn_path = Path(pgn_path)
    if source_provenance is None:
        if source_sha256 is None:
            source_sha256 = sha256_file(archive_path)
        if decompressed_sha256 is None:
            decompressed_sha256 = sha256_file(pgn_path)
        source = {
            "archive_name": archive_path.name,
            "archive_size": archive_path.stat().st_size,
            "archive_sha256": source_sha256,
            "decompressed_name": pgn_path.name,
            "decompressed_size": pgn_path.stat().st_size,
            "decompressed_sha256": decompressed_sha256,
        }
    else:
        source = {
            "archive_name": source_provenance["archive"]["name"],
            "archive_size": source_provenance["archive"]["size"],
            "archive_sha256": source_provenance["archive"]["sha256"],
            "decompressed_name": source_provenance["decompressed"]["name"],
            "decompressed_size": source_provenance["decompressed"]["size"],
            "decompressed_sha256": source_provenance["decompressed"]["sha256"],
        }
    return {
        "format_version": 3,
        "maia2_version": __version__,
        "torch_version": str(torch.__version__),
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "critical_config_sha256": _run_manifest(cfg)["critical_config_sha256"],
        "config": _config_snapshot(cfg),
        "source": source,
    }


def _validated_decompression_provenance(pgn_path, source_sha256):
    """Return verified archive/PGN provenance or fail before training."""

    pgn_path = Path(pgn_path)
    archive_path = Path(f"{pgn_path}.zst")
    provenance = read_decompression_provenance(pgn_path)
    if provenance is None:
        raise RuntimeError(
            f"Missing or invalid decompression provenance for {pgn_path}."
        )

    archive = provenance["archive"]
    decompressed = provenance["decompressed"]
    mismatches = []
    if archive.get("name") != archive_path.name:
        mismatches.append("archive name")
    if archive.get("size") != archive_path.stat().st_size:
        mismatches.append("archive size")
    if archive.get("sha256", "").lower() != source_sha256.lower():
        mismatches.append("archive SHA-256")
    if decompressed.get("name") != pgn_path.name:
        mismatches.append("decompressed name")
    if decompressed.get("size") != pgn_path.stat().st_size:
        mismatches.append("decompressed size")
    if mismatches:
        raise RuntimeError(
            f"Decompression provenance does not match {pgn_path}: "
            + ", ".join(mismatches)
            + "."
        )
    return provenance


def _file_identity(path):
    """Return a cheap identity used to detect source mutation during training."""

    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _validate_sha256_digest(value, *, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        raise ValueError(
            f"{label} must be a 64-character hexadecimal SHA-256 digest."
        ) from None
    if len(decoded) != 32:
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest.")
    return value.lower()


def _source_month(pgn_path):
    name = Path(pgn_path).name
    prefix = "lichess_db_standard_rated_"
    if not name.startswith(prefix) or not name.endswith(".pgn"):
        raise ValueError(
            f"Cannot derive a YYYY-MM source key from unexpected PGN name {name!r}."
        )
    return name.removeprefix(prefix).removesuffix(".pgn")


def _source_hash_expectations(cfg, pgn_paths):
    """Resolve and validate source hashes before model construction or training."""

    pgn_paths = list(dict.fromkeys(str(path) for path in pgn_paths))
    expected = getattr(cfg, "source_sha256", None)
    if expected is None:
        return {path: None for path in pgn_paths}

    months = {_source_month(path) for path in pgn_paths}
    if isinstance(expected, str):
        if len(months) != 1:
            raise ValueError(
                "source_sha256 may be one digest only when the configured range "
                "contains one unique month. For multiple months, provide a "
                "complete mapping keyed by YYYY-MM or source filename."
            )
        digest = _validate_sha256_digest(expected, label="source_sha256")
        return {path: digest for path in pgn_paths}

    if not isinstance(expected, dict):
        raise TypeError(
            "source_sha256 must be a SHA-256 digest string or a complete mapping "
            "keyed by YYYY-MM or source filename."
        )

    provided = {}
    for raw_key, raw_digest in expected.items():
        if not isinstance(raw_key, (str, Path)):
            raise TypeError("source_sha256 mapping keys must be strings or paths.")
        key = Path(str(raw_key)).name
        if key in provided:
            raise ValueError(f"Duplicate normalized source_sha256 key: {key!r}.")
        provided[key] = _validate_sha256_digest(
            raw_digest, label=f"source_sha256[{raw_key!r}]"
        )

    resolved = {}
    used_keys = set()
    missing = []
    for path in pgn_paths:
        filename = Path(path).name
        aliases = {_source_month(path), filename, f"{filename}.zst"}
        matches = aliases.intersection(provided)
        if not matches:
            missing.append(_source_month(path))
            continue
        if len(matches) > 1:
            raise ValueError(
                f"source_sha256 specifies the {_source_month(path)} archive more "
                f"than once via aliases: {sorted(matches)}."
            )
        key = matches.pop()
        used_keys.add(key)
        resolved[path] = provided[key]

    if missing:
        raise ValueError(
            "source_sha256 mapping is incomplete; missing configured month(s): "
            + ", ".join(sorted(missing))
        )
    unused = set(provided).difference(used_keys)
    if unused:
        raise ValueError(
            "source_sha256 mapping contains unknown key(s): "
            + ", ".join(sorted(unused))
        )
    return resolved


def _verify_expected_source_hash(cfg, pgn_path, expected=_UNSET):
    if expected is _UNSET:
        expected = _source_hash_expectations(cfg, [pgn_path])[str(pgn_path)]
    archive_path = Path(pgn_path + ".zst")
    actual = sha256_file(archive_path)
    if expected is not None and actual.lower() != expected.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {archive_path}: expected {expected}, got {actual}."
        )
    action = "Verified" if expected is not None else "Calculated"
    print(f"{action} source SHA-256: {actual}", flush=True)
    return actual


def _validate_checkpoint_metadata(
    checkpoint, cfg, *, expected_epoch, expected_archive_name, expected_source_sha256
):
    """Reject a resume checkpoint whose recorded run is incompatible."""

    metadata = checkpoint.get("training_metadata")
    if metadata is None:
        warnings.warn(
            "The resume checkpoint has no training_metadata. Loading it as a "
            "legacy checkpoint; architecture and data provenance cannot be "
            "validated automatically.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint training_metadata must be a mapping.")

    expected_critical_sha256 = _run_manifest(cfg)["critical_config_sha256"]
    recorded_critical_sha256 = metadata.get("critical_config_sha256")
    if (
        not isinstance(recorded_critical_sha256, str)
        or recorded_critical_sha256.lower() != expected_critical_sha256
    ):
        raise RuntimeError(
            "Checkpoint critical configuration SHA-256 is missing or "
            "incompatible: "
            f"{recorded_critical_sha256!r} != {expected_critical_sha256!r}."
        )

    if metadata.get("epoch") != expected_epoch:
        raise RuntimeError(
            "Checkpoint metadata epoch does not match the configured resume "
            f"epoch: {metadata.get('epoch')!r} != {expected_epoch!r}."
        )

    source = metadata.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(
            "Checkpoint training_metadata is missing its source manifest."
        )
    if source.get("archive_name") != expected_archive_name:
        raise RuntimeError(
            "Checkpoint source archive does not match the configured resume "
            f"month: {source.get('archive_name')!r} != {expected_archive_name!r}."
        )
    if expected_source_sha256 is not None:
        recorded = source.get("archive_sha256")
        if not isinstance(recorded, str) or recorded.lower() != expected_source_sha256:
            raise RuntimeError(
                "Checkpoint source SHA-256 does not match source_sha256 for the "
                f"configured resume month: {recorded!r} != "
                f"{expected_source_sha256!r}."
            )

    saved_config = metadata.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError(
            "Checkpoint training_metadata is missing its config snapshot."
        )
    saved_critical = _critical_config(saved_config)
    current_critical = _critical_config(cfg)
    mismatches = []
    for group, values in current_critical.items():
        for key, current_value in values.items():
            saved_value = saved_critical[group][key]
            if saved_value != current_value:
                mismatches.append(
                    f"{group}.{key}: checkpoint={saved_value!r}, "
                    f"current={current_value!r}"
                )
    if mismatches:
        raise RuntimeError(
            "Checkpoint critical training configuration is incompatible:\n  "
            + "\n  ".join(mismatches)
        )


def _normalize_optimizer_state_devices(optimizer):
    """Keep AdamW counters on CPU and tensor moments beside their parameters."""

    for parameter, state in optimizer.state.items():
        for key, value in tuple(state.items()):
            if not torch.is_tensor(value):
                continue
            target = torch.device("cpu") if key == "step" else parameter.device
            state[key] = value.to(target)


def _validate_optimizer_hyperparameters(optimizer, cfg):
    """Reject a checkpoint that would silently replace configured AdamW values."""

    expected = {"lr": float(cfg.lr), "weight_decay": float(cfg.wd)}
    mismatches = []
    for index, parameter_group in enumerate(optimizer.param_groups):
        for key, expected_value in expected.items():
            actual = parameter_group.get(key)
            if actual is None or float(actual) != expected_value:
                mismatches.append(
                    f"param_group[{index}].{key}: checkpoint={actual!r}, "
                    f"configured={expected_value!r}"
                )
    if mismatches:
        raise RuntimeError(
            "Checkpoint optimizer hyperparameters are incompatible:\n  "
            + "\n  ".join(mismatches)
        )


def _load_resume_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    cfg,
    *,
    expected_source_sha256=None,
):
    """Load and validate a checkpoint without staging optimizer state on a GPU."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_archive_name = (
        f"lichess_db_standard_rated_{cfg.checkpoint_year}-"
        f"{cfg.checkpoint_month:02d}.pgn.zst"
    )
    _validate_checkpoint_metadata(
        checkpoint,
        cfg,
        expected_epoch=cfg.checkpoint_epoch,
        expected_archive_name=expected_archive_name,
        expected_source_sha256=expected_source_sha256,
    )
    load_model_state_dict(model, checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _validate_optimizer_hyperparameters(optimizer, cfg)
    _normalize_optimizer_state_devices(optimizer)
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def _training_schedule(cfg, pgn_paths):
    """Return zero-based epochs and month paths for a fresh or resumed run.

    A checkpoint is written after a complete month. Resuming therefore skips
    that month and all earlier months only in the checkpoint's epoch, then
    returns to the complete configured month range for later epochs.
    """

    if cfg.max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer.")
    if not pgn_paths:
        raise ValueError("The configured training range contains no usable months.")

    if not cfg.from_checkpoint:
        return [(epoch, list(pgn_paths)) for epoch in range(cfg.max_epochs)]

    checkpoint_epoch = cfg.checkpoint_epoch
    if not 1 <= checkpoint_epoch <= cfg.max_epochs:
        raise ValueError(
            "checkpoint_epoch must be between 1 and max_epochs when resuming."
        )

    checkpoint_month = (
        f"lichess_db_standard_rated_{cfg.checkpoint_year}-"
        f"{cfg.checkpoint_month:02d}.pgn"
    )
    try:
        checkpoint_index = [Path(path).name for path in pgn_paths].index(
            checkpoint_month
        )
    except ValueError:
        raise ValueError(
            "The checkpoint month must be present in the configured training "
            "range. Keep the original full start/end range when resuming."
        ) from None

    first_epoch = checkpoint_epoch - 1
    schedule = []
    for epoch in range(first_epoch, cfg.max_epochs):
        epoch_paths = list(pgn_paths)
        if epoch == first_epoch:
            epoch_paths = epoch_paths[checkpoint_index + 1 :]
        schedule.append((epoch, epoch_paths))
    return schedule


def run(cfg, device="auto"):
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
    seed_everything(cfg.seed)
    device = resolve_device(device)
    print(f"\tdevice: {device}", flush=True)
    num_processes = get_num_processes(cfg.num_cpu_left)

    pgn_paths = read_monthly_data_path(cfg)
    training_schedule = _training_schedule(cfg, pgn_paths)
    source_hashes = _source_hash_expectations(cfg, pgn_paths)

    base_save_root = Path(getattr(cfg, "save_root", "../saves")).expanduser()
    save_root = base_save_root / f"{cfg.lr}_{cfg.batch_size}_{cfg.wd}"
    save_root.mkdir(parents=True, exist_ok=True)
    overwrite_checkpoints = getattr(cfg, "overwrite_checkpoints", False)
    manifest_path = save_root / _RUN_MANIFEST_NAME
    defer_legacy_manifest = cfg.from_checkpoint and not manifest_path.exists()
    if not defer_legacy_manifest:
        _ensure_run_manifest(save_root, cfg)
    _validate_checkpoint_destinations(
        save_root,
        training_schedule,
        overwrite_checkpoints=overwrite_checkpoints,
    )

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)

    print(model, flush=True)
    model = model.to(device)
    if _should_use_data_parallel(device):
        model = nn.DataParallel(model)
    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N_params = count_parameters(model)
    print(f"Trainable Parameters: {N_params}", flush=True)

    accumulated_samples = 0
    accumulated_games = 0

    if cfg.from_checkpoint:
        formatted_month = f"{cfg.checkpoint_month:02d}"
        checkpoint_path = save_root / (
            f"epoch_{cfg.checkpoint_epoch}_"
            f"{cfg.checkpoint_year}-{formatted_month}.pgn.pt"
        )
        checkpoint_pgn_name = (
            f"lichess_db_standard_rated_{cfg.checkpoint_year}-{formatted_month}.pgn"
        )
        checkpoint_pgn_path = next(
            path for path in pgn_paths if Path(path).name == checkpoint_pgn_name
        )
        checkpoint = _load_resume_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            cfg,
            expected_source_sha256=source_hashes[checkpoint_pgn_path],
        )
        if defer_legacy_manifest:
            # Only claim a legacy directory after its requested checkpoint has
            # loaded and passed every validation available to that format.
            _ensure_run_manifest(save_root, cfg, allow_legacy_resume=True)
        accumulated_samples = checkpoint["accumulated_samples"]
        accumulated_games = checkpoint["accumulated_games"]

    for epoch, epoch_pgn_paths in training_schedule:
        print(f"Epoch {epoch + 1}", flush=True)
        if not epoch_pgn_paths:
            print(
                "The resumed checkpoint already completed the final month of "
                "this epoch; continuing to the next epoch.",
                flush=True,
            )
            continue

        num_file = 0
        for pgn_path in epoch_pgn_paths:
            start_time = time.time()
            source_sha256 = _verify_expected_source_hash(
                cfg, pgn_path, source_hashes[pgn_path]
            )
            decompress_zst(
                pgn_path + ".zst",
                pgn_path,
                reuse_existing=getattr(cfg, "reuse_decompressed", False),
            )
            provenance = _validated_decompression_provenance(pgn_path, source_sha256)
            decompressed_sha256 = provenance["decompressed"]["sha256"]
            pgn_identity = _file_identity(pgn_path)
            print(
                f"Decompressing {pgn_path} took {readable_time(time.time() - start_time)}",
                flush=True,
            )

            pgn_chunks = read_or_create_chunks(
                pgn_path,
                cfg,
                source_fingerprint=decompressed_sha256,
            )
            print(f"Training {pgn_path} with {len(pgn_chunks)} chunks.", flush=True)

            if not pgn_chunks:
                _require_trained_month(
                    pgn_path,
                    positions=0,
                    games=0,
                    optimizer_steps=0,
                )

            pgn_chunks_sublists = []
            for i in range(0, len(pgn_chunks), num_processes):
                pgn_chunks_sublists.append(pgn_chunks[i : i + num_processes])

            num_chunk = 0
            month_positions = 0
            month_games = 0
            optimizer_steps_before = _optimizer_step_count(optimizer)
            for data, game_count, chunk_count in _iter_preprocessed_batches(
                cfg,
                pgn_path,
                pgn_chunks_sublists,
                elo_dict,
            ):
                num_chunk += chunk_count
                month_games += game_count
                accumulated_games += game_count
                if not data:
                    print(
                        f"[{num_chunk}/{len(pgn_chunks)}] No positions passed "
                        "the training filters; skipping this chunk batch.",
                        flush=True,
                    )
                    continue

                loss, loss_maia, loss_side_info, loss_value = train_chunks(
                    cfg,
                    data,
                    model,
                    optimizer,
                    all_moves_dict,
                    criterion_maia,
                    criterion_side_info,
                    criterion_value,
                )
                month_positions += len(data)
                accumulated_samples += len(data)
                print(f"[{num_chunk}/{len(pgn_chunks)}]", flush=True)
                print(f"[# Positions]: {readable_num(accumulated_samples)}", flush=True)
                print(f"[# Games]: {readable_num(accumulated_games)}", flush=True)
                print(
                    f"[# Loss]: {loss} | [# Loss MAIA]: {loss_maia} | [# Loss Side Info]: {loss_side_info} | [# Loss Value]: {loss_value}",
                    flush=True,
                )

            if num_chunk != len(pgn_chunks):
                raise RuntimeError(
                    f"Preprocessed {num_chunk} of {len(pgn_chunks)} PGN chunks."
                )

            month_optimizer_steps = (
                _optimizer_step_count(optimizer) - optimizer_steps_before
            )
            _require_trained_month(
                pgn_path,
                positions=month_positions,
                games=month_games,
                optimizer_steps=month_optimizer_steps,
            )
            if _file_identity(pgn_path) != pgn_identity:
                raise RuntimeError(
                    f"Decompressed source changed during training: {pgn_path}. "
                    "No checkpoint will be written."
                )
            print(
                f"Month totals: positions={month_positions}, "
                f"games={month_games}, optimizer_steps={month_optimizer_steps}",
                flush=True,
            )

            num_file += 1
            print(
                f"### ({num_file} / {len(epoch_pgn_paths)}) Took {readable_time(time.time() - start_time)} to train {pgn_path} with {len(pgn_chunks)} chunks.",
                flush=True,
            )
            _save_checkpoint_atomic(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accumulated_samples": accumulated_samples,
                    "accumulated_games": accumulated_games,
                    "training_metadata": _training_metadata(
                        cfg,
                        pgn_path,
                        epoch + 1,
                        _optimizer_step_count(optimizer),
                        source_sha256=source_sha256,
                        decompressed_sha256=decompressed_sha256,
                        source_provenance=provenance,
                    ),
                    "rng_state": _capture_rng_state(device),
                },
                save_root / _checkpoint_name(epoch + 1, pgn_path),
                overwrite=overwrite_checkpoints,
            )
            Path(pgn_path).unlink()
            decompression_provenance_path(pgn_path).unlink(missing_ok=True)
