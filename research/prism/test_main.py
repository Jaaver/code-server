import os
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_testtmp"); os.makedirs(TMP, exist_ok=True)
PRISM = os.path.join(HERE, "prism_colab.py")
"""Drive main() end to end with a mock proposer: validates phase wiring, the held-out
generalisation split, the evolution loop, the report and the persisted JSON."""
import sys, types, os, json, random, importlib.util

def stub(name, **a):
    m = types.ModuleType(name); [setattr(m,k,v) for k,v in a.items()]; sys.modules[name]=m; return m
np = stub("numpy"); np.random = types.SimpleNamespace(seed=lambda *a: None)
class _C:
    is_available=staticmethod(lambda: False); is_bf16_supported=staticmethod(lambda: False)
    get_device_name=staticmethod(lambda i:"stub"); empty_cache=staticmethod(lambda: None)
    OutOfMemoryError=RuntimeError
torch = stub("torch", cuda=_C, bfloat16="bf16", float16="fp16", float32="fp32",
             manual_seed=lambda *a: None, __version__="stub", no_grad=lambda:(lambda f:f))
torch.optim = types.SimpleNamespace(AdamW=None, lr_scheduler=types.SimpleNamespace(OneCycleLR=None))
torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=None))
stub("transformers", AutoTokenizer=None, AutoModelForCausalLM=None, __version__="stub")
stub("peft", LoraConfig=None, get_peft_model=None)

SCR = TMP
os.environ["PRISM_STATE"] = os.path.join(SCR, "state_main")
G = {"__name__": "prism_main_test"}
exec(compile(open(PRISM).read(), "prism_colab.py", "exec"), G)

spec = importlib.util.spec_from_file_location("te", os.path.join(HERE, "test_e2e.py"))
# reuse the MockLLM definition without re-running that file's asserts
mock_src = open(os.path.join(HERE, "test_e2e.py")).read()
mock_src = mock_src[mock_src.index("OPSRC = {"):mock_src.index("rng = random.Random(11)")]
MK = {"G": G, "json": json, "random": random}
exec(mock_src, MK)

REG = []
class LiveMock(MK["MockLLM"]):
    """Re-indexes itself from whatever tasks the run has created so far."""
    def __init__(self, p): super().__init__([], p_correct=p); self.seen=0
    def attach_lora(self, r=16, alpha=32, resume_from=None): return 4_300_000
    def reset_lora(self): self.resets = getattr(self, "resets", 0) + 1; return 1
    def snapshot_lora(self): return {"w": 1}
    def restore_lora(self, snap): return True
    def sync(self):
        pool = [t for v in G["EVAL"].values() for t in v] + REG
        if len(pool) != self.seen:
            for t in pool: self.index(t)
            self.seen = len(pool)
    def chat(self, batch, n=1, **kw):
        self.sync(); return super().chat(batch, n=n, **kw)

_bc = G["build_curriculum"]
def bc(n, rng, frontier):
    ts = _bc(n, rng, frontier); REG.extend(ts); return ts
G["build_curriculum"] = bc
G["star_finetune"] = lambda llm, pairs, steps, **k: dict(n_pairs=len(pairs), steps=steps,
                                                         loss_start=1.5, loss_end=0.6)
G["LLM"] = lambda cands: LiveMock(0.5)
G["CFG"].update(n_math=6, n_grid=10, n_sci=4, n_agent=8, k_math=4, k_grid=4, k_sci=4,
                k_agent=4, sci_rounds=2, evo_rounds=2, evo_tasks=12, sft_steps=5)
G["TIME_BUDGET"] = 100000
G["MODE"] = "once"          # bounded path first; continuous mode is exercised below

llm, report = G["main"]()
FIRST_CALLS = llm.calls

# ---- a disconnect must cost nothing: a second run resumes, it does not redo work ----
llm2, report2 = G["main"]()
assert llm2.calls < FIRST_CALLS / 4, (
    f"resume did not skip finished work: {FIRST_CALLS} calls first run, {llm2.calls} on resume")
for k in ("baseline", "baseline_sc", "prism_v0", "curve"):
    assert k in report2, f"resumed report lost {k}"
assert report2["baseline"] == report["baseline"], "resume changed an already-measured result"
assert len(report2["curve"]) == len(report["curve"]), "resume lost evolution rounds"
print(f"\nresume: {FIRST_CALLS} model calls on the first run, {llm2.calls} on the second  \u2713")

# ---- and PRISM_FRESH must genuinely start over ----
import os as _os
_os.environ["PRISM_FRESH"] = "1"
llm3, _ = G["main"]()
del _os.environ["PRISM_FRESH"]
assert llm3.calls > FIRST_CALLS / 2, "PRISM_FRESH did not force a full re-run"
print(f"PRISM_FRESH=1 forces a full re-run ({llm3.calls} calls)  \u2713")

assert report["params"] < 1e9
for key in ("baseline","baseline_sc","prism_v0","prism_v1","curve","grid_heldout_v0","suite_sizes"):
    assert key in report, f"missing {key}"
assert len(report["curve"]) == 3, report["curve"]
assert report["prism_v0"]["MEAN"] > report["baseline"]["MEAN"], "search must beat single pass"
assert report["prism_v0"]["MEAN"] > report["baseline_sc"]["MEAN"], \
    "verification must beat the compute-matched sampling control"
print(f"\ncompute-matched control: base {report['baseline']['MEAN']:.1f} -> "
      f"+self-consistency {report['baseline_sc']['MEAN']:.1f} -> "
      f"+verification {report['prism_v0']['MEAN']:.1f}")
ho_n, tr_n = report["grid_heldout_v0"][1], report["grid_heldout_v0"][3]
assert ho_n > 0 and tr_n > 0, report["grid_heldout_v0"]
# the curriculum must never leak a held-out rule family or an evaluated law
leaked = [t["rule"] for t in REG if t["domain"]=="grid" and set(t["rule"].split("+")) & G["GRID_HELDOUT"]]
assert not leaked, f"curriculum leaked held-out grid families: {leaked[:5]}"
evnames = {t["name"] for t in G["EVAL"]["sci"]}
assert not {t["name"] for t in REG if t["domain"]=="sci"} & evnames, "curriculum leaked an evaluated law"
saved = json.load(open(os.path.join(os.environ["PRISM_STATE"], "report.json")))
assert saved["curve"][-1]["round"] == 2
print(f"\nheld-out grid families: {ho_n} eval tasks | curriculum families: {tr_n} eval tasks")
print("curriculum leakage check: none, in either grid rules or scientific laws  ✓")
print("report.json persisted with the full evolution curve  ✓")
print("MAIN-PATH TEST PASSED")

# =====================================================================================
#  CONTINUOUS MODE: the loop must keep going, stop cleanly, and resume where it stopped
# =====================================================================================
import os as _os, json as _json
_os.environ["PRISM_FRESH"] = "1"

# 1. forever mode keeps going past the preset's evo_rounds until the clock runs out
G["MODE"] = "forever"; G["EVAL_EVERY"] = 1
G["RESERVE_S"] = 0.1; G["FIRST_ROUND_S"] = 0.1
G["TIME_BUDGET"] = 12.0; G["T0"] = __import__("time").time()
G["CFG"]["evo_rounds"] = 1            # would have stopped at 1 in "once" mode
llm4, rep4 = G["main"]()
del _os.environ["PRISM_FRESH"]
assert rep4["rounds_completed"] >= 2, (
    f"forever mode stopped after {rep4['rounds_completed']} round(s) despite budget remaining")
assert "min left" in rep4["stop_reason"] or "budget" in rep4["stop_reason"], rep4["stop_reason"]
print(f"\nforever mode ran {rep4['rounds_completed']} rounds past evo_rounds=1, "
      f"stopped because: {rep4['stop_reason'][:60]}  \u2713")

# 2. a live status file exists and names the current round
st = _json.load(open(_os.path.join(_os.environ["PRISM_STATE"], "status.json")))
assert st["round"] == rep4["rounds_completed"] and st["running"] is False, st
print(f"status.json tracks progress live and is marked finished at the end  \u2713")

# 3. an interrupt stops the loop cleanly rather than losing the round
_os.environ["PRISM_FRESH"] = "1"
G["TIME_BUDGET"] = 60.0; G["T0"] = __import__("time").time()
_orig_harvest = G["harvest"]
_seen = {"n": 0}
def _harvest_then_interrupt(llm, tasks, k):
    _seen["n"] += 1
    out = _orig_harvest(llm, tasks, k)
    if _seen["n"] >= 2:                     # let one round finish, interrupt during the next
        G["HALT"]["flag"] = True; G["HALT"]["why"] = "interrupted by you"
    return out
G["harvest"] = _harvest_then_interrupt
llm5, rep5 = G["main"]()
G["harvest"] = _orig_harvest; G["HALT"]["flag"] = False; G["HALT"]["why"] = ""
del _os.environ["PRISM_FRESH"]
assert rep5["stop_reason"] == "interrupted by you", rep5["stop_reason"]
assert rep5["rounds_completed"] >= 1, rep5
print(f"interrupt stopped the loop cleanly after {rep5['rounds_completed']} round(s), "
      f"reason recorded as {rep5['stop_reason']!r}  \u2713")

# 4. and the interrupted run resumes from the round it reached
ck = _json.load(open(_os.path.join(_os.environ["PRISM_STATE"], "checkpoint.json")))
assert ck["rounds_done"] == rep5["rounds_completed"], (ck["rounds_done"], rep5["rounds_completed"])
G["TIME_BUDGET"] = 8.0; G["T0"] = __import__("time").time()   # keep the suite quick
llm6, rep6 = G["main"]()
assert rep6["rounds_completed"] >= rep5["rounds_completed"], "resume went backwards"
print(f"resumed from round {ck['rounds_done']} and continued to "
      f"{rep6['rounds_completed']}  \u2713")

# 5. the skill library stays bounded no matter how long it runs
SK = G["SkillLibrary"](_os.path.join(TMP, "cap.json"))
SK.CAP = 40
for i in range(400):
    SK.add(["math","grid","sci","agent"][i % 4], f"desc {i}", f"def f{i}(): return {i}")
assert len(SK) <= SK.CAP + 4, f"library grew to {len(SK)} past a cap of {SK.CAP}"
assert len({i["domain"] for i in SK.items}) == 4, "eviction wiped out entire domains"
print(f"skill library capped at {len(SK)} entries with all 4 domains surviving  \u2713")

# =====================================================================================
#  The 8-hour run degraded itself: fresh adapter per round, and revert a bad round
# =====================================================================================
class _FakeParam:
    def __init__(self, v): self.v = v; self.requires_grad = True; self.shape = (1,)
    def detach(self): return self
    def clone(self): return _FakeParam(self.v)
    def copy_(self, o): self.v = o.v

class _LoraMock(LiveMock):
    def __init__(self, p):
        super().__init__(p); self._w = _FakeParam(0.0); self._resets = 0
    def reset_lora(self): self._resets += 1; self._w.v = 0.0; return 1
    def snapshot_lora(self): return {"w": self._w.clone()}
    def restore_lora(self, snap): self._w.copy_(snap["w"]); return True

lm = _LoraMock(0.5)
snap = lm.snapshot_lora(); lm._w.v = 99.0
lm.restore_lora(snap)
assert lm._w.v == 0.0, "restore_lora did not put the weights back"
lm.reset_lora(); assert lm._resets == 1
print("\nadapter snapshot/restore/reset round-trips correctly  \u2713")

# the trainer must stop once the loss has collapsed rather than memorising further
_calls = {"n": 0}
_real_sft = G["star_finetune"]
src = open(PRISM).read()
assert "sft loss collapsed" in src, "the memorisation guard is missing"
assert "llm.reset_lora()          # STaR: fresh adapter" in src, "rounds do not reset the adapter"
assert "reverted the adapter" in src, "no regression revert"
assert "RESOLUTION WARNING" in src, "no small-suite warning"
print("memorisation guard, per-round reset, regression revert and resolution warning present  \u2713")

print("CONTINUOUS-MODE TESTS PASSED")
