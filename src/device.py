"""Device selection for training and explanation runs.

The pipeline was written CPU-only. Nothing in it moved tensors to an
accelerator, and several call sites went straight from a tensor to
`.numpy()`, which raises on a CUDA tensor. This module centralises the
choice so the config drives it (CONVENTIONS.md §4: no hardcoded
hyperparameters in `src/`) rather than each script guessing.
"""

import torch


def resolve_device(config: dict) -> torch.device:
    """Pick the torch device for this run.

    Args:
        config: Parsed experiment config. Reads the optional top-level
            `device` key: "auto" (default) picks CUDA when available and
            falls back to CPU, or name a device explicitly ("cpu", "cuda",
            "cuda:0") to pin it.

    Returns:
        The torch.device to run on.
    """
    spec = str(config.get("device", "auto")).lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def describe_device(device: torch.device) -> str:
    """Human-readable one-liner for run logs.

    Args:
        device: The device returned by resolve_device().

    Returns:
        e.g. "cuda:0 (Tesla T4, 15.8 GB)" or "cpu".
    """
    if device.type != "cuda":
        return str(device)
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    return f"{device} ({props.name}, {props.total_memory / 2**30:.1f} GB)"
