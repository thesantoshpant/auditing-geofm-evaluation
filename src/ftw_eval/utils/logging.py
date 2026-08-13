from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def get_logger(name: str = "ftw_eval", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def banner(msg: str, char: str = "=") -> None:
    line = char * max(60, len(msg) + 4)
    print(line, file=sys.stderr)
    print(f" {msg}", file=sys.stderr)
    print(line, file=sys.stderr)
