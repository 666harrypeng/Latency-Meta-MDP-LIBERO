"""Repository asset root shared by command and library code."""

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]
