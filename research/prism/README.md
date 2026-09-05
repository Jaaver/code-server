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
                      ├─ domain tools (below)
                      └─ skill retrieval from the library below
```

### What PRISM is given that the controls are not

This is the crux of reading the table honestly. PRISM is not the same prompt with more
sampling — it is the same weights inside a scaffold. The scaffold contributes:

| tool | what it does | why it is legitimate |
|---|---|---|
| sandbox | runs every candidate and feeds the traceback back | the model still has to write the program |
| exact verifiers | decide correctness mechanically | they cannot be talked into a wrong answer — an always-wrong proposer scores exactly 0, and `test_e2e` asserts it |
| constant fitting (`sci`) | least-squares fits a scale and offset around a proposed skeleton | the standard skeleton+optimiser split in symbolic regression; the model must still supply the form |
| scaling analysis (`sci`) | log-log regression of the data, passed into the prompt | what a physicist does first; recovers exponents for power laws and says "not a power law" otherwise, in deliberately generic terms so no benchmark answer is named |
| grid rendering | lays pairs out as grids, not one-line lists | presentation, not information — the literals are shown too |
| invariant inference (`grid`) | structural facts true of every example: shape relation, whether the value multiset is preserved, whether rows or columns are merely reordered | invariant inference is standard in program synthesis; it narrows the hypothesis class without naming any rule |
| skill library | retrieves the system's own past verified solutions | its own work, earned under the same verifier |

The single-pass and self-consistency controls get none of this. That is the point: the table
measures the scaffold, and the scaffold is the claim.

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
| `PRISM_THINK` | `auto` | `auto` uses a chat template's reasoning mode when it has one, `0` forces it off, `1` forces it on. Reasoning tokens are test-time compute, so this trades wall-clock for quality |
| `PRISM_FRONTIER_KEY` / `_BASE` / `_MODEL` | unset | run a real frontier head-to-head on the same prompts and the same grader, no tools |
| `PRISM_FRONTIER_K` | `1` | give the frontier model the same self-consistency budget as the control row |

Any OpenAI-compatible endpoint works (`_BASE`). Calls that error out are excluded from the
frontier's score and reported, never counted as wrong answers — an unreachable API must not be
able to flatter this system.

The run asserts `n_params < 1e9` and aborts if the constraint is violated.

## Measured limits

Testing against real sub-1B weights (Qwen2.5-0.5B-Instruct, Qwen2.5-Coder-0.5B-Instruct and
Qwen3-0.6B, on CPU) found a hard boundary worth stating up front, because the run prints
per-suite verdicts that will show it to you:

* **`math` works.** The stack solved multi-step problems 2/2 where a single greedy pass scored
  0 — the model can write a correct program, and the sandbox plus voting finds it. Measurement
  also showed program-of-thought *alone* falling below plain chain-of-thought at low `k` on
  GSM8K, because the model has to parse the story before it can write the program; the solver
  therefore samples both formats and pools them into one vote, weighting an executed program
  above a hand-computed answer.
* **`grid` does not.** Across four prompt designs, two model families, grid-block rendering,
  invariant inference and `k = 8`, every configuration scored 0/3 on rules as simple as
  `flipud`. The proposals were never close, so there was nothing for the verifier to select.

That asymmetry is the most useful thing this repo measures: **a verifier selects, it cannot
invent.** Where the model's sample set contains a correct program, verification converts a
failing model into a working solver. Where it does not, no amount of test-time compute helps,
and the suite stays at zero. Search multiplies a proposer that is sometimes right; it does
nothing for one that is never right.

The `grid` suite is deliberately kept in the benchmark for exactly this reason. Dropping the
suite the system fails would make the table look better and mean less.

## Running without a GPU

The script detects a missing GPU and shrinks the suites and sample counts so a CPU run still
finishes, but a CPU run is a smoke test, not a measurement: with a quarter of the items and
half the samples, one task is worth 25 points and the vote has almost nothing to weigh. A
verified CPU run of the full pipeline took 54 minutes and scored zero across the board while
the single-pass baseline scored 8.3 — and the report said so, in those words, rather than
rounding it up. Use a T4.

## What this is not

It is not a superintelligence, it does not beat frontier models or humans in general, and the
self-evolution loop saturates. Its advantage is confined to problems whose answers can be checked
mechanically. Remove the verifier and it is a 0.6B model again. The script prints this same
caveat next to its own results.
