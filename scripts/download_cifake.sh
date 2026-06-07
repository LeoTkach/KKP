#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
TARGET="${ROOT}/data/ai_generated"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install: pip install kaggle" >&2
  echo "Then place API token at ~/.kaggle/kaggle.json" >&2
  exit 1
fi

mkdir -p "${TARGET}"

kaggle datasets download \
  birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p "${TARGET}" \
  --unzip

# Kaggle may unpack into a nested folder — flatten if needed
NESTED="${TARGET}/cifake-real-and-ai-generated-synthetic-images"
if [[ -d "${NESTED}/train" ]]; then
  mv "${NESTED}/train" "${TARGET}/"
  mv "${NESTED}/test" "${TARGET}/"
  rmdir "${NESTED}" 2>/dev/null || rm -rf "${NESTED}"
fi

echo "CIFAKE ready at ${TARGET}"
echo "Expected layout:"
echo "  ${TARGET}/train/REAL"
echo "  ${TARGET}/train/FAKE"
echo "  ${TARGET}/test/REAL"
echo "  ${TARGET}/test/FAKE"
