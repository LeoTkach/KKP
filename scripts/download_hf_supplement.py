#!/usr/bin/env python3
"""Deprecated — use scripts/download_ai_hires.py instead."""

from __future__ import annotations

import sys


def main() -> None:
    print("Deprecated. Run: python scripts/download_ai_hires.py", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
