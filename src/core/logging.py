"""
TrafficVision-AI :: Structured Logging
========================================
Enterprise-grade, JSON-structured, context-aware logging.
Compatible with ELK, Datadog, GCP Cloud Logging, AWS CloudWatch.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Request-scoped context propagation
# ---------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_user_id: ContextVar[str] = ContextVar("user_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def set_request_context(
    request_id: str = "",
    user_id: str = "",
    trace_id: str = "",
) -> None:
    _request_id.set(request_id or str(uuid.uuid4()))
    _user_id.set(user_id)
    _trace_id.set(trace_id or str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log record — machine-parseable and ELK-ready."""

    SERVICE_NAME = "trafficvision-ai"
    SERVICE_VERSION = "2.0.0"

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.SERVICE_NAME,
            "version": self.SERVICE_VERSION,
            "environment": "unknown",
            "request_id": _request_id.get(""),
            "user_id": _user_id.get(""),
            "trace_id": _trace_id.get(""),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge any extra fields attached by the caller
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[union-attr]

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> "StructuredLogger":
    return StructuredLogger(name, level)


class StructuredLogger:
    """Thin wrapper that adds structured key-value pairs to every log call."""

    def __init__(self, name: str, level: str = "INFO") -> None:
        self._log = logging.getLogger(name)
        self._log.setLevel(getattr(logging, level.upper(), logging.INFO))

        if not self._log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self._log.addHandler(handler)
            self._log.propagate = False

    def _emit(
        self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        record = self._log.makeRecord(
            self._log.name,
            level,
            "(unknown)",
            0,
            msg,
            (),
            None,
        )
        record.extra = extra or {}  # type: ignore[attr-defined]
        self._log.handle(record)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._log.exception(msg, extra={"extra": kwargs})

    @contextmanager
    def timed(self, operation: str, **context: Any):
        """Context manager that logs duration of a block."""
        start = time.perf_counter()
        self.info(f"{operation} started", **context)
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.error(
                f"{operation} failed",
                duration_ms=round(elapsed * 1000, 2),
                error=str(exc),
                **context,
            )
            raise
        else:
            elapsed = time.perf_counter() - start
            self.info(
                f"{operation} completed",
                duration_ms=round(elapsed * 1000, 2),
                **context,
            )


# ---------------------------------------------------------------------------
# Configure root logging on import
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        root.addHandler(handler)
