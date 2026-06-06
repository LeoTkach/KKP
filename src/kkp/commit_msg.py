"""Перевірка формату commit message (Conventional Commits)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "ci",
    "build",
    "perf",
    "revert",
)

COMMIT_MESSAGE_PATTERN = re.compile(
    rf"^({'|'.join(ALLOWED_TYPES)})(\([\w.\-/]+\))?!?: .+\S$",
)

IGNORE_PATTERNS = (
    re.compile(r"^Merge "),
    re.compile(r"^Revert "),
    re.compile(r"^fixup! "),
    re.compile(r"^squash! "),
)

HELP = f"""
Повідомлення коміту не відповідає Conventional Commits.

Формат: <type>(<scope>): <description>

Типи: {", ".join(ALLOWED_TYPES)}

Приклади:
  feat: add data loader
  fix: correct image preprocessing
  docs: update readme
  chore: init project
  test: add config loader tests
  ci: add docker build step
"""


def read_commit_header(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def should_ignore(header: str) -> bool:
    return any(pattern.match(header) for pattern in IGNORE_PATTERNS)


def validate(header: str) -> bool:
    if not header:
        print("Порожнє повідомлення коміту.", file=sys.stderr)
        return False

    if should_ignore(header):
        return True

    if COMMIT_MESSAGE_PATTERN.match(header):
        return True

    print(HELP.strip(), file=sys.stderr)
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m kkp.commit_msg <commit-msg-file>", file=sys.stderr)
        return 1

    header = read_commit_header(Path(sys.argv[1]))
    return 0 if validate(header) else 1


if __name__ == "__main__":
    raise SystemExit(main())
