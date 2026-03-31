"""
Run all ``scripts.figure*`` modules in the order listed below.
From the repository root: ``python main.py``
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os

_ROOT = Path(__file__).resolve().parent

FIGURE_SCRIPTS: list[str] = [
    "scripts.figure2a",
    "scripts.figure2b",
    "scripts.figure3a",
    "scripts.figure3b",
    "scripts.figure3c",
    "scripts.figure4a",
    "scripts.figure4b",
    "scripts.figure4c",
    "scripts.figure5a",
    "scripts.figure5b",
    "scripts.figure5c",
]


def main() -> None:
    for name in FIGURE_SCRIPTS:
        print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}")
        subprocess.run([sys.executable, "-m", name], cwd=str(_ROOT), check=True)


if __name__ == "__main__":
    main()
