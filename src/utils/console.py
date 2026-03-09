"""console.py
Simple terminal formatting helpers.
"""

from typing import Iterable


def format_box(lines: str | Iterable[str]) -> str:
    """Render one or more lines inside a plain box with solid borders."""
    if isinstance(lines, str):
        normalized = [lines]
    else:
        normalized = [str(line) for line in lines]

    if not normalized:
        normalized = [""]

    width = max(len(line) for line in normalized)
    top = "┌" + "─" * (width + 2) + "┐"
    body = [f"│ {line.ljust(width)} │" for line in normalized]
    bottom = "└" + "─" * (width + 2) + "┘"
    return "\n".join([top, *body, bottom])


def print_box(lines: str | Iterable[str]) -> None:
    """Print one or more lines inside a plain box, followed by a blank line."""
    print(format_box(lines))
    print()
