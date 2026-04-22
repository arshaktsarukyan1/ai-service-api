import logging
import sys


class _StructuredFormatter(logging.Formatter):
    """Single-line structured log formatter: timestamp level logger message [k=v ...]."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras: list[str] = []
        skip = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                extras.append(f"{key}={value!r}")
        return f"{base} {' '.join(extras)}" if extras else base


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured single-line format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _StructuredFormatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.root.setLevel(level.upper())
    logging.root.handlers = [handler]
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
