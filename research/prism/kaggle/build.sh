#!/usr/bin/env bash
# Assemble a pushable Kaggle kernel: an environment preamble followed by the whole engine.
# Kaggle takes exactly one code file and accepts no environment variables, so anything that
# would be an env var elsewhere is baked into the top of the generated file.
#
#   ./build.sh OUTDIR [SLUG] [KEY=VALUE ...]
#   ./build.sh /tmp/k                                    # continuous run, tiny preset
#   ./build.sh /tmp/k prism-measure PRISM_PRESET=measure PRISM_MODE=once
set -e -o pipefail
cd "$(dirname "$0")"
OUT="${1:-.}"; shift || true
SLUG="${1:-prism-sub1b-continuous}"; shift || true
mkdir -p "$OUT"

TITLE=$(echo "$SLUG" | tr '-' ' ')
sed "s/prism-sub1b-continuous/$SLUG/; s/PRISM sub1b continuous/$TITLE/" \
    kernel-metadata.json > "$OUT/kernel-metadata.json"

{
  echo "# PRISM on Kaggle. Save & Run All (Commit) queues this server-side: close the browser,"
  echo "# it keeps going and stops itself with a report."
  echo "#   Settings > Accelerator: GPU     Settings > Internet: On  (needed for weights)"
  echo "import os"
  echo 'os.environ.setdefault("PRISM_MODE", "forever")'
  echo 'os.environ.setdefault("PRISM_PRESET", "tiny")'
  echo 'os.environ.setdefault("PRISM_MAX_HOURS", "8.0")'
  echo 'os.environ.setdefault("PRISM_DRIVE", "0")   # /kaggle/working already persists'
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    echo "os.environ[\"$k\"] = \"$v\""
  done
  cat ../prism_colab.py
} > "$OUT/prism_run.py"

# compile(), not ast.parse(): only compile() enforces rules like "__future__ must come first",
# which is exactly how the first pushed version passed review here and failed on Kaggle.
python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" "$OUT/prism_run.py"
echo "built $OUT/prism_run.py ($(wc -l < "$OUT/prism_run.py") lines) slug=$SLUG"
grep -E '^os\.environ' "$OUT/prism_run.py" | sed 's/^/    /'
