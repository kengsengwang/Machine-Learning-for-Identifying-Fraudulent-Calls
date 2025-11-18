# src/utils.py
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a simple console logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
