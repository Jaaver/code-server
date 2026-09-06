# =====================================================================================
#  Diagnostic probe, not a benchmark run.
#
#  The 8-hour T4 run scored 0.0 on grid, sci and agent in all 71 evaluations, and solved
#  0 of 280 self-generated planning tasks. The conclusion drawn was that a sub-1B proposer
#  never puts a correct program in the candidate set. But one lever was never actually
#  tested: Qwen3-0.6B's reasoning mode. Earlier attempts gave it 900 tokens, which the CPU
#  logs showed was not enough to finish a single thought, so that test proved nothing.
#
#  This asks exactly one question, cheaply: with enough tokens to finish thinking, does
#  reasoning move grid and agent off zero? It sweeps the token budget so the answer
#  distinguishes "the model cannot do this" from "I never let it finish a sentence".
# =====================================================================================
import time, random, json, collections

llm = LLM(["Qwen/Qwen3-0.6B"] + MODEL_CANDIDATES)
llm.tok_mult = 1.0                      # budgets below are absolute
log("probe model: %s | reasoning template available: %s",
    llm.model_id, llm._detect_thinking() if True else "?")

rng = random.Random(11)
GRID = gen_grid(8, rng, difficulty=1)
AGENT = [t for t in gen_agent(16, rng) if t["tier"] <= 2][:6]
log("probe suites: %d grid (%s), %d agent (tiers %s)",
    len(GRID), ",".join(t["rule"] for t in GRID[:4]) + ",...",
    len(AGENT), sorted({t["tier"] for t in AGENT}))

def run_trial(label, think, maxtok, k=4):
    """Returns (finished%, parsed%, solved%) per suite. 'finished' means the reply survived
    _strip_think, i.e. the model actually closed its reasoning block within the budget."""
    llm.think = think
    out = {}
    for kind, tasks in (("grid", GRID), ("agent", AGENT)):
        t0 = time.time(); fin = tot = par = 0; solved = 0
        prompts = [(_grid_prompt(t) if kind == "grid" else _agent_prompt(t)) for t in tasks]
        cands = llm.chat(prompts, n=k, max_new_tokens=maxtok, temperature=0.9)
        for t, cs in zip(tasks, cands):
            hit = False
            for c in cs:
                tot += 1
                if c.strip(): fin += 1
                else: continue
                code = extract_code(c)
                if kind == "grid":
                    if "def transform" in code:
                        par += 1
                        pred, _ = _grid_check(code, t)
                        if pred == t["test"][1]: hit = True
                else:
                    mv = ""
                    if "def solve" in code:
                        par += 1
                        h = (strip_to_definitions(code, "solve") + "\ngrid=" + json.dumps(t["grid"]) +
                             "\nprint(''.join(x for x in str(solve(grid)).upper() if x in 'UDLR'))\n")
                        r = run_python(h, timeout=10)
                        mv = (r["out"].splitlines() or [""])[-1].strip() if r["ok"] else ""
                    else:
                        mv = _plan_from_text(c)
                        if mv: par += 1
                    if mv and agent_simulate(t["grid"], mv, t["max_steps"])[0]: hit = True
            solved += int(hit)
        out[kind] = dict(finished=100.0*fin/max(1,tot), parsed=100.0*par/max(1,tot),
                         solved=100.0*solved/max(1,len(tasks)), secs=time.time()-t0)
        log("  %-34s %-6s finished %5.1f%% | parsed %5.1f%% | SOLVED %5.1f%%  (%.0fs)",
            label, kind, out[kind]["finished"], out[kind]["parsed"], out[kind]["solved"],
            out[kind]["secs"])
    return out

rule("REASONING PROBE — does letting it think move grid/agent off zero?")
TRIALS = [("no reasoning, 400 tok  (control)", False, 400),
          ("reasoning, 800 tok",               True,  800),
          ("reasoning, 1600 tok",              True,  1600),
          ("reasoning, 2800 tok",              True,  2800)]
results = {}
for label, think, mx in TRIALS:
    if budget_left() < 600:
        log("out of budget before %s", label); break
    results[label] = run_trial(label, think, mx)
    json.dump(results, open(os.path.join(STATE_DIR, "probe.json"), "w"), indent=2)

rule("PROBE RESULT")
hdr = f"{'condition':<36}{'grid fin':>9}{'grid solved':>12}{'agent fin':>10}{'agent solved':>13}"
print(hdr); print("-" * len(hdr))
for label in results:
    g, a = results[label]["grid"], results[label]["agent"]
    print(f"{label:<36}{g['finished']:>8.0f}%{g['solved']:>11.0f}%"
          f"{a['finished']:>9.0f}%{a['solved']:>12.0f}%")
print("-" * len(hdr))
best = max((r["grid"]["solved"] for r in results.values()), default=0.0)
besta = max((r["agent"]["solved"] for r in results.values()), default=0.0)
print(f"\nbest grid {best:.0f}%  best agent {besta:.0f}%   (both were 0% for 8 hours on the 0.5B)")
if best == 0 and besta == 0:
    print("VERDICT: reasoning tokens do not rescue these suites. The candidate set genuinely")
    print("never contains a correct program, so verification has nothing to select. That is")
    print("the real ceiling, not a budget artefact.")
else:
    print("VERDICT: reasoning moves it off zero — the earlier zeros were partly a token-budget")
    print("artefact, and the full run should use this mode at the budget that worked.")
json.dump(results, open(os.path.join(STATE_DIR, "probe.json"), "w"), indent=2)
log("probe.json written to %s", STATE_DIR)
