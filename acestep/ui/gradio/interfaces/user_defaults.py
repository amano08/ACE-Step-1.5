"""Environment-driven startup defaults for the generation tab.

Lets a launch script preset the caption, negative prompt and the optional
metadata fields (BPM / key / time signature / duration) without editing UI code.

All helpers fall back to the stock UI behaviour when the variable is unset, so
an unconfigured checkout behaves exactly as before.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _clean(name: str) -> str | None:
    """Return a stripped env var value, or ``None`` when unset/blank."""
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def remix_only() -> bool:
    """Whether the UI should expose only the Remix (cover) workflow.

    When ``ACESTEP_UI_REMIX_ONLY`` is truthy the generation tab is built in the
    Remix configuration and every control that does not affect a ``cover`` task
    is hidden. Components are hidden rather than removed so the existing event
    wiring keeps resolving.
    """
    value = _clean("ACESTEP_UI_REMIX_ONLY")
    return value is not None and value.lower() not in ("0", "false", "no", "off")


def caption_default(fallback_file: Path) -> str:
    """Read the startup caption from ``ACESTEP_DEFAULT_CAPTION_FILE`` or a fallback path.

    Args:
        fallback_file: Path used when the env var is unset.

    Returns:
        The caption text, or an empty string when no file is readable.
    """
    override = _clean("ACESTEP_DEFAULT_CAPTION_FILE")
    candidate = Path(override) if override else fallback_file
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    try:
        return candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def negative_prompt_default(stock: str) -> str:
    """Return the startup LM negative prompt.

    ``ACESTEP_DEFAULT_NEGATIVE_PROMPT_FILE`` takes precedence over
    ``ACESTEP_DEFAULT_NEGATIVE_PROMPT``. Anything equal to the stock sentinel is
    treated as "no negative prompt" by the LM (see
    ``LLMHandler._has_meaningful_negative_prompt``).

    Args:
        stock: The upstream default sentinel, normally ``"NO USER INPUT"``.

    Returns:
        The negative prompt text to prefill.
    """
    path_override = _clean("ACESTEP_DEFAULT_NEGATIVE_PROMPT_FILE")
    if path_override:
        candidate = Path(path_override)
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / candidate
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text
    return _clean("ACESTEP_DEFAULT_NEGATIVE_PROMPT") or stock


def bpm_default() -> float | None:
    """Return the startup BPM from ``ACESTEP_DEFAULT_BPM``, or ``None`` for auto."""
    value = _clean("ACESTEP_DEFAULT_BPM")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def keyscale_default() -> str:
    """Return the startup key/scale from ``ACESTEP_DEFAULT_KEYSCALE``, or ``""`` for auto."""
    return _clean("ACESTEP_DEFAULT_KEYSCALE") or ""


def timesig_default() -> str:
    """Return the startup time signature from ``ACESTEP_DEFAULT_TIMESIG``, or ``""`` for auto.

    Accepts either the bare beat count the dropdown uses (``"3"``) or a ``"3/4"``
    style string, which is reduced to its numerator.
    """
    value = _clean("ACESTEP_DEFAULT_TIMESIG")
    if value is None:
        return ""
    numerator = value.split("/")[0].strip()
    return numerator or ""


def duration_default() -> float | None:
    """Return the startup duration in seconds from ``ACESTEP_DEFAULT_DURATION``, or ``None`` for auto."""
    value = _clean("ACESTEP_DEFAULT_DURATION")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_env(name: str) -> float | None:
    """Return an env var parsed as float, or ``None`` when unset or unparseable."""
    value = _clean(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inference_steps_default() -> int | None:
    """Return the startup DiT step count from ``ACESTEP_DEFAULT_STEPS``, or ``None``.

    ``None`` keeps the per-model value chosen by ``get_ui_control_config`` (50 for
    SFT, 8 for turbo, 32 otherwise).
    """
    value = _float_env("ACESTEP_DEFAULT_STEPS")
    return int(value) if value is not None else None


def guidance_scale_default() -> float | None:
    """Return the startup CFG scale from ``ACESTEP_DEFAULT_GUIDANCE_SCALE``, or ``None``."""
    return _float_env("ACESTEP_DEFAULT_GUIDANCE_SCALE")


def audio_cover_strength_default(stock: float) -> float:
    """Return the startup remix strength from ``ACESTEP_DEFAULT_REMIX_STRENGTH``.

    Args:
        stock: Value used when the env var is unset.
    """
    value = _float_env("ACESTEP_DEFAULT_REMIX_STRENGTH")
    return stock if value is None else value


def cover_noise_strength_default(stock: float) -> float:
    """Return the startup cover strength from ``ACESTEP_DEFAULT_COVER_STRENGTH``.

    Args:
        stock: Value used when the env var is unset.
    """
    value = _float_env("ACESTEP_DEFAULT_COVER_STRENGTH")
    return stock if value is None else value


def _audio_path(name: str) -> str | None:
    """Resolve an audio path env var against the repo root.

    Returns ``None`` when unset or when the file no longer exists, so a moved or
    deleted preset leaves an empty upload box instead of breaking the UI build.
    """
    value = _clean(name)
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return str(candidate) if candidate.is_file() else None


def reference_audio_default() -> str | None:
    """Return the startup reference audio from ``ACESTEP_DEFAULT_REFERENCE_AUDIO``.

    The reference is style conditioning on the encoder side. ``process_reference_audio``
    samples 3x10s windows, and only a file of exactly 30s (or one that loops to
    exactly 30s) makes that sampling deterministic, so preset clips should be 30s.
    """
    return _audio_path("ACESTEP_DEFAULT_REFERENCE_AUDIO")


def source_audio_default() -> str | None:
    """Return the startup source audio from ``ACESTEP_DEFAULT_SOURCE_AUDIO``.

    This is the track a ``cover`` task is conditioned on, uploaded every session
    otherwise.
    """
    return _audio_path("ACESTEP_DEFAULT_SOURCE_AUDIO")
