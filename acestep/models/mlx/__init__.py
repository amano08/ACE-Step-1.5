# Native MLX implementations of AceStep models for Apple Silicon.
# Provides pure MLX inference with graceful fallback to PyTorch.

import logging
import platform

logger = logging.getLogger(__name__)


class _LoguruBridge(logging.Handler):
    """Forward this package's stdlib records to loguru.

    Every module under ``acestep.models.mlx`` logs through ``logging``, but the
    app only configures loguru sinks, so the root logger keeps its default
    WARNING level and drops INFO records entirely -- including the
    ``[MLX-DiT] Cover mode: ...`` diagnostic that reports how many diffusion
    steps actually survive ``cover_noise_strength`` truncation.  Attaching the
    bridge to the package logger (not the root) keeps third-party stdlib
    logging untouched.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from loguru import logger as _loguru
        except ImportError:  # pragma: no cover - loguru is a hard dependency
            return
        # opt(depth=6) walks past the logging machinery so loguru reports the
        # original call site; exception=... preserves tracebacks.
        _loguru.opt(depth=6, exception=record.exc_info).log(
            record.levelname, record.getMessage()
        )


if not any(isinstance(h, _LoguruBridge) for h in logger.handlers):
    logger.addHandler(_LoguruBridge())
    logger.setLevel(logging.INFO)
    # Records are delivered to loguru here; letting them also reach the root
    # logger would double-print once anything configures a root handler.
    logger.propagate = False


def is_mlx_available() -> bool:
    """Check if MLX is available on this platform (macOS + Apple Silicon)."""
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core as mx
        import mlx.nn
        # Verify we can actually create arrays (Metal backend works)
        _ = mx.array([1.0])
        mx.eval(_)
        return True
    except Exception:
        return False


_MLX_AVAILABLE = None


def mlx_available() -> bool:
    """Cached check for MLX availability."""
    global _MLX_AVAILABLE
    if _MLX_AVAILABLE is None:
        _MLX_AVAILABLE = is_mlx_available()
    return _MLX_AVAILABLE
