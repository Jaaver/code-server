# PRISM — a sub-1B reasoning engine that trades parameters for search, verification and memory

`prism_colab.py` is a single self-contained Google Colab cell. It loads a **<1B-parameter**
open-weight model, wraps it in a verifier-guided reasoning stack, benchmarks it against the
*same weights* used the ordinary way, then lets it **improve itself with no human labels** and
re-benchmarks after every round.

```
Runtime > Change runtime type > T4 GPU     (CPU works too, much slower)
paste the whole file into one cell, run it
```

### Where to run it if you want to close everything

Colab keeps a runtime alive only while the tab is open and the machine is awake, unless you
have Pro+ background execution. Three alternatives that genuinely run detached:

| | cost | how |
|---|---|---|
| **Kaggle Notebooks** | free, roughly 30 GPU-hours/week | paste the script in, Accelerator: GPU, Internet: **On**, then **Save & Run All (Commit)**. That queues it server-side — close the browser, the output is waiting when you come back. `/kaggle/working` is detected automatically and persists as notebook output. |
| **Modal** | usage-based, has a free monthly credit allowance | `modal run --detach modal_app.py` — returns immediately, job keeps running. State goes in a Modal Volume so re-runs resume. Watch with `modal app logs prism`. |
| **RunPod / Vast.ai / Lambda** | hourly, cheapest per GPU-hour | rent a box, `tmux new -s prism`, run `python prism_colab.py`, detach with `Ctrl-B D`, close everything. Remember to stop the box — it bills while it exists. |

Quotas and pricing on all of these move; check current terms rather than trusting this table.

### Continuous operation

The default is `PRISM_MODE=forever`: after the initial benchmark it keeps evolving round
after round — new curriculum, verify, distil into skills and weights, re-benchmark — until
one of three things happens.

* **You interrupt it.** `SIGINT`/`SIGTERM` set a stop flag rather than killing the process:
  the round in flight finishes, state is checkpointed, the report is printed. Signal twice to
  force-quit.
* **The host stops it.** Kaggle's `SIGTERM` at the session limit is handled the same way.
* **The clock runs out.** `PRISM_MAX_HOURS` (default 8.5, sized to fit inside a Kaggle GPU
  session) and the loop refuses to start a round it cannot finish, using the measured duration
  of recent rounds, keeping back enough time to write the report.

Every round is checkpointed as it completes, so re-running always continues from the last
finished round. `status.json` is rewritten after each round with the live round number, skill
count, best score and recent curve, so you can see progress mid-run without attaching to it.

Over a long run the curriculum ramps on its own — grid difficulty and the minimum agent tier
both rise once the system's own pass rate on them clears 60% — and the skill library is capped
per domain so it cannot grow without bound or let one easy domain crowd out the rest. If the
score stops improving for three evaluations the log says so plainly; it keeps running, but it
does not pretend saturation is progress.

**A disconnect costs you nothing.** Colab kills a runtime when the tab closes or the machine
sleeps, and `/content` dies with it — so state goes to Google Drive (one mount click on the
first run) and every phase checkpoints as it finishes. If the runtime drops, re-run the same
cell: it prints what it is skipping and picks up from the last finished phase. `PRISM_FRESH=1`
starts over deliberately.

The default preset is `tiny` (~12-20 min on a T4) for the same reason. A run you have to sit
and watch for an hour is a worse experiment than a short one you can actually finish, and the
longer presets are there when you want them, not by default.

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
| `PRISM_PRESET` | `tiny` | `tiny` (~12-20 min per pass) / `quick` / `standard` / `full` |
| `PRISM_MODE` | `forever` | `forever` evolves until stopped; `once` does a single benchmark-and-evolve pass |
| `PRISM_MAX_HOURS` | `8.5` | wall-clock ceiling in `forever` mode |
| `PRISM_EVAL_EVERY` | preset | re-benchmark every Nth round; the expensive half of a round |
| `PRISM_SKILL_CAP` | `600` | maximum skill-library entries before per-domain eviction |
| `PRISM_DRIVE` | `1` | mount Google Drive so state survives the runtime dying; `0` keeps it in `/content` |
| `PRISM_FRESH` | `0` | `1` ignores the checkpoint and re-runs every phase |
| `PRISM_TIME_BUDGET` | preset | seconds; every phase degrades gracefully instead of hanging |
| `PRISM_MODEL` | auto | forces a base model; the loader otherwise walks a fallback chain |
| `PRISM_THINK` | `auto` | `auto` uses a chat template's reasoning mode when it has one, `0` forces it off, `1` forces it on. Reasoning tokens are test-time compute, so this trades wall-clock for quality |
| `PRISM_FRONTIER_KEY` / `_BASE` / `_MODEL` | unset | run a real frontier head-to-head on the same prompts and the same grader, no tools |
| `PRISM_FRONTIER_K` | `1` | give the frontier model the same self-consistency budget as the control row |

Any OpenAI-compatible endpoint works (`_BASE`). Calls that error out are excluded from the
frontier's score and reported, never counted as wrong answers — an unreachable API must not be
able to flatter this system.

The run asserts `n_params < 1e9` and aborts if the constraint is violated.

## Result of the first full run (Kaggle T4, 8 hours, 70 rounds)

`results/kaggle-t4-run1-report.json` is the real thing, on a real T4, with LoRA working.
It is a negative result and it is kept here on purpose.

```
system                                 math   grid    sci  agent    MEAN
base 494M, 1 pass, no tools            40.0    0.0    0.0    0.0    10.0
base + self-consistency, no tools      50.0    0.0    0.0    0.0    12.5
PRISM v0 (search+verify)               30.0    0.0    0.0    0.0     7.5
```

* **The scaffold lost.** On the only suite that scored at all, PRISM came in below both
  controls. On a 10-item suite those are 3, 4 and 5 problems — the difference is noise, but
  there is certainly no evidence of the gain the design predicts.
* **grid, sci and agent read 0.0 in all 71 evaluations — but that reading was mostly an
  artefact.** The curriculum measured the same solvers over far more tasks, and the true rates
  are not zero: grid 23/350 = 6.6%, sci 28/350 = 8.0%, math 84/350 = 24.0%. An 8-item suite
  facing a 6.6% solver returns exactly zero **58% of the time**, and a 3-item sci suite returns
  zero 78% of the time. The suites were too small to distinguish "cannot" from "rarely".
  Only **agent is genuinely zero**: 0 of 280, which is a real ceiling.

  Resolving these rates to ±3 points needs roughly 263 grid items, 315 sci and 779 math. The
  `measure` preset exists for that; `tiny` cannot answer the question it appears to answer.
* **Self-evolution ran backwards.** Math averaged 42.0 over the first 35 rounds and 26.1 over
  the last 35. The cause is visible in the logs: LoRA training loss hit 0.000 on the second
  update and stayed under 0.01 for 59 of 67 updates. The adapter memorised its 133 verified
  traces and was then retrained on them 67 more times.

Three fixes went in because of this run: each round now resets the adapter and retrains from
scratch on the accumulated traces (which is what STaR actually specifies); training stops when
the loss collapses; and a round that scores materially below the best reverts to the
best-known weights, so an unattended loop cannot walk downhill for eight hours. The run also
exposed that `tiny` prints a results table nobody should read as a measurement, so the report
now says so itself.

## Second and third runs: the powered head-to-head, and the reasoning probe

Both are in `results/`. Both landed on a **P100**, which this torch build has no kernels for,
so both fell back to CPU with quarter-size suites. The fallback is why the numbers exist at
all rather than being silent zeros — but it also means neither ran at the size or speed asked
for, and between them they consumed roughly 12 of a 30-hour weekly GPU allowance doing CPU
work on GPU-billed sessions. The script now aborts instead (`PRISM_REQUIRE_GPU`).

**The head-to-head, pooled with run 1** (37 + 10 math items):

| | baseline | + self-consistency | PRISM |
|---|---|---|---|
| run 1 (tiny, real T4, n=10) | 40.0% | 50.0% | 30.0% |
| run 2 (measure, CPU, n=37) | 40.5% | 40.5% | 29.7% |
| **pooled, n=47** | **40.4%** | — | **29.8%** |

PRISM is **10.6 points below the plain single pass**, two-proportion z = 1.08 — not
significant, but the same direction in two independent runs. The mechanism is visible in the
earlier CPU measurement: program-of-thought alone scores below plain chain-of-thought on
GSM8K for this model, because it has to parse the story before it can write the program, and
a confidently misread story produces a confidently wrong program that the executed-answer
vote then weights *above* the prose answer. The scaffold's core mechanism is net-negative on
the only suite where this model can do anything.

Grid produced its first non-zero reading anywhere: **1/37 = 2.7%** for PRISM against 0/37 for
the baseline, consistent with the 6.6% curriculum rate.

**The reasoning probe** swept Qwen3-0.6B (751.6M params) at 400 tokens without reasoning and
800/1600 with:

| condition | grid: thought finished | grid solved | agent finished | agent solved |
|---|---|---|---|---|
| no reasoning, 400 tok | 100% | 0% | 100% | 0% |
| reasoning, 800 tok | 3% | 0% | 0% | 0% |
| reasoning, 1600 tok | 19% | 0% | 0% | 0% |

Read this carefully: at 1600 tokens only 19% of samples closed their reasoning block, so the
solved column rests on about six usable answers. It is **not** clean evidence that reasoning
cannot help — it is evidence that reasoning at a budget this model needs is impractical here.
The 1600-token grid trial alone took 3.2 hours, and the 2800-token trial never ran.

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
