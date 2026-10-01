"""Cooperative stop flag for an in-flight generation run.

Gradio's ``cancels=`` releases the UI, but the worker thread already inside a
blocking diffusion call keeps computing until it returns.  This module lets the
Stop button raise the flag and the diffusion loop abort at its next step.
"""

import threading

_stop_event = threading.Event()


class GenerationStopped(Exception):
    """Raised inside a generation loop when the user pressed Stop."""


def request_stop() -> None:
    """Ask running generation loops to abort at their next checkpoint."""
    _stop_event.set()


def clear_stop() -> None:
    """Clear a previous stop request so the next run is not aborted."""
    _stop_event.clear()


def raise_if_stopped() -> None:
    """Abort the calling generation loop when a stop has been requested."""
    if _stop_event.is_set():
        raise GenerationStopped("Generation stopped by user")
