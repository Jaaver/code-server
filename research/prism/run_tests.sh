#!/usr/bin/env bash
# Offline test suite: no GPU, no network, no model download.
# test_core  - generators, verifiers, sandbox, skill library, parsers
# test_e2e   - solvers/harvest/evaluate against oracle, noisy and adversarial proposers
# test_main  - the whole main() path, held-out split, curriculum leakage, report JSON
set -e
# pipefail matters here: without it `python3 t.py | tail` reports tail's exit status, so a
# crashing test printed "ALL SUITES PASSED". It did, once.
set -o pipefail
cd "$(dirname "$0")"
rm -rf _testtmp
for t in test_core test_e2e test_main; do
  echo "===== $t ====="
  python3 "$t.py" | tail -6
done
echo "ALL SUITES PASSED"
