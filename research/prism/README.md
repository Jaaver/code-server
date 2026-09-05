# PRISM — a sub-1B reasoning engine that trades parameters for search, verification and memory

`prism_colab.py` is a single self-contained Google Colab cell. It loads a **<1B-parameter**
open-weight model, wraps it in a verifier-guided reasoning stack, benchmarks it against the
*same weights* used the ordinary way, then lets it **improve itself with no human labels** and
re-benchmarks after every round.

```
Runtime > Change runtime type > T4 GPU     (CPU works too, much slower)
paste the whole file into one cell, run it
```

## The claim, stated precisely

A 0.6B model cannot out-think a 1T frontier model. What it *can* do is win on problems where
**checking an answer is cheap and exact**, because there a small model becomes a *proposal
distribution* and correctness is decided by an executable verifier, not by the model. Search
then substitutes for scale. Everything in this repo is built to make that claim measurable
rather than rhetorical:

* every benchmark has an exact, mechanical ground-truth checker;
* there are **two** controls, not one — the identical weights on a single greedy pass, and a
  **compute-matched** control that gets the same `k` samples with majority voting but no
  program execution and no verifier. The gap between those two is what sampling buys; the gap
  above them is what *verification* buys, and it is much larger;
* four grid rule families and every evaluated scientific law are withheld from the
  self-evolution curriculum, so generalisation is reported separately from in-distribution gain;
* `PRISM_FRONTIER_KEY` runs a real frontier model through the *identical* protocol, so the
  head-to-head is measured rather than asserted.

The printed table is therefore a ladder:

```
base 494M, 1 pass, no tools      <- what the weights do unaided
base + self-consistency, no tools <- + k samples, majority vote  (compute-matched)
PRISM v0 (search+verify)          <- + programs, execution, exact verification, refinement
PRISM v1..vN (self-evolved)       <- + skills and LoRA distilled from its own verified work
FRONTIER <model> (1 pass)         <- optional, identical protocol
```

## The four suites (all exactly verifiable)

| suite | task | verifier |
|---|---|---|
| `math` | GSM8K test split, or a procedural multi-step word-problem generator when offline | exact numeric match |
| `grid` | ARC-lite induction: infer a grid transformation from 3 examples, emit `transform(g)` | must reproduce all examples, then exact-match a held-out grid |
| `sci`  | recover a closed-form physical law from raw measurements | normalised MSE over 140 rows; `<1e-8` counts as exact symbolic recovery. PRISM may propose the *skeleton* and have a scale and offset least-squares fitted for it (the standard split in symbolic regression, and what makes a 0.6B proposer useful); the single-pass controls are scored without that fit, so the tool is never credited to the raw model |
| `agent`| goal-directed planning in a grid world with ordered items, keys and doors, 4 difficulty tiers | the returned action string is simulated step by step |

Each generator is validated against a reference oracle, and each verifier is validated against
an adversarial dud proposer that must score exactly zero.

## The stack

```
                      ┌─ program synthesis (the model writes Python, not prose)
                      ├─ sandboxed execution (subprocess, timeout, isolated cwd)
sub-1B base model ────┼─ exact verification (the four checkers above)
                      ├─ self-consistency voting over k samples
                      ├─ error-driven refinement (the failure message goes back in)
                      └─ skill retrieval from the library below
```

## Self-evolution — two channels of memory, zero human labels

1. **Auto-curriculum.** The system invents its own tasks and raises difficulty in any domain
   where its own pass rate exceeds 60%.
2. **Verifier-filtered harvest.** Only trajectories an exact checker accepts survive.
3. **Non-parametric memory.** Survivors become entries in an executable *skill library*
   (`skills.json`), retrieved by TF-IDF into later prompts. This persists across sessions, so
   the system is strictly stronger the second time you run it.
4. **Parametric memory.** The same survivors are distilled back into the weights by masked-prompt
   LoRA SFT — rejection-sampling fine-tuning, i.e. STaR.
5. Re-benchmark, repeat. The printed curve is that loop running.

`evolve_more(3)` in a later cell continues the loop; state lives in `/content/prism_state`.

## Configuration

| env var | default | meaning |
|---|---|---|
| `PRISM_PRESET` | `quick` | `quick` / `standard` / `full` |
| `PRISM_TIME_BUDGET` | preset | seconds; every phase degrades gracefully instead of hanging |
| `PRISM_MODEL` | auto | forces a base model; the loader otherwise walks a fallback chain |
| `PRISM_FRONTIER_KEY` / `_BASE` / `_MODEL` | unset | run a real frontier head-to-head on the same prompts and the same grader, no tools |
| `PRISM_FRONTIER_K` | `1` | give the frontier model the same self-consistency budget as the control row |

Any OpenAI-compatible endpoint works (`_BASE`). Calls that error out are excluded from the
frontier's score and reported, never counted as wrong answers — an unreachable API must not be
able to flatter this system.

The run asserts `n_params < 1e9` and aborts if the constraint is violated.

## What this is not

It is not a superintelligence, it does not beat frontier models or humans in general, and the
self-evolution loop saturates. Its advantage is confined to problems whose answers can be checked
mechanically. Remove the verifier and it is a 0.6B model again. The script prints this same
caveat next to its own results.
