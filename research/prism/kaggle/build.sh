#!/usr/bin/env bash
# Assemble the pushable kernel: an environment preamble followed by the whole script.
# Kaggle kernels take exactly one code file and accept no environment variables, so the
# settings that would be env vars elsewhere are baked into the top of the file here.
set -e
cd "$(dirname "$0")"
OUT="${1:-.}"
mkdir -p "$OUT"
cp kernel-metadata.json "$OUT/"
{
  cat <<'PY'
# ---------------------------------------------------------------------------
# PRISM on Kaggle. Save & Run All (Commit) queues this server-side: close the
# browser, it keeps evolving and stops itself with a report.
#   Settings > Accelerator: GPU     Settings > Internet: On  (needed for weights)
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PRISM_MODE", "forever")
os.environ.setdefault("PRISM_PRESET", "tiny")
os.environ.setdefault("PRISM_MAX_HOURS", "8.0")   # inside Kaggle's GPU session limit
os.environ.setdefault("PRISM_DRIVE", "0")         # /kaggle/working persists as output
PY
  cat ../prism_colab.py
} > "$OUT/prism_run.py"
# compile(), not ast.parse(): only compile() enforces rules like "__future__ must come
# first", which is exactly how the first pushed version got through review and failed on
# Kaggle instead of here.
python3 -c "import sys; src=open(sys.argv[1]).read(); compile(src, sys.argv[1], 'exec')" \
    "$OUT/prism_run.py"
echo "built and compiled $OUT/prism_run.py ($(wc -l < "$OUT/prism_run.py") lines)"
