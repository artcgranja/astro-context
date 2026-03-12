"""Shared callback-firing utility used by pipeline and memory subsystems."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any


def fire_callbacks(
    callbacks: Sequence[Any],
    method: str,
    *args: Any,
    logger: logging.Logger | None = None,
    log_level: int = logging.WARNING,
    **kwargs: Any,
) -> None:
    """Fire a method on all callbacks, swallowing exceptions.

    Parameters:
        callbacks: Sequence of callback objects to notify.
        method: Name of the method to call on each callback.
        *args: Positional arguments forwarded to the callback method.
        logger: Optional logger for recording failures.
        log_level: Log level for failure messages (default ``WARNING``).
        **kwargs: Keyword arguments forwarded to the callback method.
    """
    for cb in callbacks:
        fn = getattr(cb, method, None)
        if fn is None or not callable(fn):
            continue
        try:
            fn(*args, **kwargs)
        except Exception:
            if logger:
                logger.log(log_level, "Callback %r.%s failed", cb, method, exc_info=True)
