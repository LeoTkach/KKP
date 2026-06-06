#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
MSG_FILE="${1:?commit message file required}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  exec "${ROOT}/.venv/bin/python" -m kkp.commit_msg "${MSG_FILE}"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m kkp.commit_msg "${MSG_FILE}"
fi

echo "Python not found. Run: make install" >&2
exit 1
