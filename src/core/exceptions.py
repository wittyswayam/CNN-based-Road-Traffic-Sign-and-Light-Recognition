"""
TrafficVision-AI :: Exception Hierarchy
=========================================
All domain-specific exceptions with structured metadata.
Maps cleanly to HTTP status codes for the REST layer.
"""

from __future__ import annotations
from http import HTTPStatus
from typing import Any, Dict, Optional


class TrafficVisionError(Exception):
    """Base exception for all TrafficVision-AI errors."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# ML / Model errors
# ---------------------------------------------------------------------------

class ModelNotFoundError(TrafficVisionError):
    http_status = HTTPStatus.NOT_FOUND.value
    error_code = "MODEL_NOT_FOUND"


class ModelLoadError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "MODEL_LOAD_FAILED"


class InferenceError(TrafficVisionError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "INFERENCE_FAILED"


class ModelDriftDetectedError(TrafficVisionError):
    http_status = HTTPStatus.SERVICE_UNAVAILABLE.value
    error_code = "MODEL_DRIFT_DETECTED"


class TrainingError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "TRAINING_FAILED"


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

class DataValidationError(TrafficVisionError):
    http_status = HTTPStatus.BAD_REQUEST.value
    error_code = "DATA_VALIDATION_FAILED"


class DataIngestionError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "DATA_INGESTION_FAILED"


class AnnotationParseError(TrafficVisionError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "ANNOTATION_PARSE_ERROR"


class ImageProcessingError(TrafficVisionError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "IMAGE_PROCESSING_FAILED"


# ---------------------------------------------------------------------------
# API / Auth errors
# ---------------------------------------------------------------------------

class AuthenticationError(TrafficVisionError):
    http_status = HTTPStatus.UNAUTHORIZED.value
    error_code = "AUTHENTICATION_FAILED"


class AuthorizationError(TrafficVisionError):
    http_status = HTTPStatus.FORBIDDEN.value
    error_code = "AUTHORIZATION_FAILED"


class RateLimitExceededError(TrafficVisionError):
    http_status = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "RATE_LIMIT_EXCEEDED"


class PayloadTooLargeError(TrafficVisionError):
    http_status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value
    error_code = "PAYLOAD_TOO_LARGE"


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------

class CacheError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "CACHE_ERROR"


class StorageError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "STORAGE_ERROR"


class ConfigurationError(TrafficVisionError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "CONFIGURATION_ERROR"
