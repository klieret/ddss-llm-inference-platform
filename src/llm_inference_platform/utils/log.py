from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import colorlog


def get_logger(
    name: str = "llmip",
    *,
    stream_level: int = logging.INFO,
    log_path: Path | None = None,
) -> logging.Logger:
    """Sets up global logger."""
    _log: logging.Logger = colorlog.getLogger(name)

    if _log.handlers:
        # the logger already has handlers attached to it, even though
        # we didn't add it ==> logging.get_logger got us an existing
        # logger ==> we don't need to do anything
        return _log

    _log.setLevel(logging.DEBUG)

    if log_path is not None:
        # Add a file handler to write log messages to a file
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        _log.addHandler(fh)

    # Add a stream handler to write log messages to the console
    sh = colorlog.StreamHandler()
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    #  This is not the same as just setting name="" in the fct arguments.
    #  This would set the root logger to debug mode, which for example causes
    #  the matplotlib font manager (which uses the root logger) to throw lots of
    #  messages. Here, we want to keep our named logger, but just drop the
    #  name.
    name_incl = "" if name == "gnn-tracking" else f" {name}"
    formatter = colorlog.ColoredFormatter(
        f"%(log_color)s[%(asctime)s{name_incl}] %(levelname)s: %(message)s",
        log_colors=log_colors,
        datefmt="%H:%M:%S",
    )
    sh.setFormatter(formatter)
    # Controlled by overall logger level
    sh.setLevel(stream_level)

    _log.addHandler(sh)

    return _log


def get_default_logger_path() -> Path:
    """Get path for default logger"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(f"llmip-{timestamp}.log")


DEFAULT_LOGGER_PATH = get_default_logger_path()

logger = get_logger(log_path=DEFAULT_LOGGER_PATH)
