"""
TensorFlow runtime configuration utilities.

Goal:
- Prefer GPU if available
- Fall back to CPU if no GPU
- Avoid common GPU OOM behavior by enabling memory growth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger


@dataclass(frozen=True)
class TensorFlowRuntimeInfo:
    device: str  # "gpu" or "cpu"
    gpus: List[str]
    note: Optional[str] = None


def configure_tensorflow(tf) -> TensorFlowRuntimeInfo:
    """
    Configure TensorFlow to use GPU when present, otherwise CPU.

    This is safe to call multiple times; it will best-effort configure devices.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        logger.warning(f"TensorFlow runtime: could not list GPUs ({e}); using CPU")
        return TensorFlowRuntimeInfo(device="cpu", gpus=[], note=str(e))

    if not gpus:
        logger.info("TensorFlow runtime: no GPU detected; using CPU")
        return TensorFlowRuntimeInfo(device="cpu", gpus=[], note="no_gpu")

    gpu_names: List[str] = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            # This can fail if TF already initialized devices; continue anyway.
            logger.warning(f"TensorFlow runtime: could not enable memory growth for {gpu} ({e})")
        gpu_names.append(getattr(gpu, "name", str(gpu)))

    # If multiple GPUs exist, TF will typically use all by default; we just log.
    logger.info(f"TensorFlow runtime: GPU detected; using GPU(s): {', '.join(gpu_names)}")
    return TensorFlowRuntimeInfo(device="gpu", gpus=gpu_names)


