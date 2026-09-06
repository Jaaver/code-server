#!/usr/bin/env bash
# Assemble the diagnostic probe kernel: settings preamble + the engine (with main()
# suppressed) + the probe itself. One file, because that is all a Kaggle kernel takes.
set -e -o pipefail
cd "$(dirname "$0")"
OUT="${1:-.}"; mkdir -p "$OUT"
sed 's/prism-sub1b-continuous/prism-reasoning-probe/; s/PRISM sub1b continuous/PRISM reasoning probe/; s/prism_run.py/probe_run.py/' \
    kernel-metadata.json > "$OUT/kernel-metadata.json"
{
  cat <<'PY'
import os
os.environ.setdefault("PRISM_NO_MAIN", "1")     # reuse the engine, do not start a run
os.environ.setdefault("PRISM_MODE", "once")
os.environ.setdefault("PRISM_PRESET", "tiny")
os.environ.setdefault("PRISM_MAX_HOURS", "2.5") # a probe, not an 8-hour run
os.environ.setdefault("PRISM_TIME_BUDGET", "9000")
os.environ.setdefault("PRISM_DRIVE", "0")
os.environ.setdefault("PRISM_THINK", "0")       # each trial sets this itself
PY
  cat ../prism_colab.py
  cat probe_reasoning.py
} > "$OUT/probe_run.py"
python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" "$OUT/probe_run.py"
echo "built and compiled $OUT/probe_run.py ($(wc -l < "$OUT/probe_run.py") lines)"
