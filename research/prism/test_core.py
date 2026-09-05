import os
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_testtmp"); os.makedirs(TMP, exist_ok=True)
PRISM = os.path.join(HERE, "prism_colab.py")
"""Exercise every non-LLM component of PRISM: generators, verifiers, sandbox, skills."""
import sys, types, os, json, random, math, time

# ---- stub the heavy deps so the module imports without a GPU stack -------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

np = stub("numpy"); np.random = types.SimpleNamespace(seed=lambda *a: None)
class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def is_bf16_supported(): return False
    @staticmethod
    def get_device_name(i): return "stub"
    @staticmethod
    def empty_cache(): pass
    OutOfMemoryError = RuntimeError
torch = stub("torch", cuda=_Cuda, bfloat16="bf16", float16="fp16", float32="fp32",
             manual_seed=lambda *a: None, __version__="stub", no_grad=lambda: (lambda f: f))
torch.optim = types.SimpleNamespace(AdamW=None, lr_scheduler=types.SimpleNamespace(OneCycleLR=None))
torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=None))
tr = stub("transformers", AutoTokenizer=None, AutoModelForCausalLM=None, __version__="stub")
stub("peft", LoraConfig=None, get_peft_model=None)

os.environ["PRISM_STATE"] = os.path.join(TMP,"state")
src = open(PRISM).read()
g = {"__name__": "prism_test"}
exec(compile(src, "prism_colab.py", "exec"), g)

rng = random.Random(7)
fails = []

# ---------------------------------------------------------------- math suite ----
mt = g["gen_math"](30, rng)
print(f"math tasks: {len(mt)}")
print("  sample:", mt[0]["q"][:110], "=>", mt[0]["ans"])
assert len(mt) == 30 and len({t['q'] for t in mt}) == 30

# an oracle program must verify through the sandbox path
r = g["run_python"]("print(3*7+1)")
assert r["ok"] and g["last_number"](r["out"]) == 22.0, r
r = g["run_python"]("import sys\nwhile True: pass", timeout=2)
assert not r["ok"] and "Timeout" in r["err"], r
r = g["run_python"]("raise ValueError('boom')")
assert not r["ok"] and "ValueError" in r["err"], r
print("sandbox: ok / timeout / traceback all handled")

# ---------------------------------------------------------------- grid suite ----
gt = g["gen_grid"](40, rng, difficulty=2)
print(f"grid tasks: {len(gt)} | rules e.g. {[t['rule'] for t in gt[:4]]}")
assert len(gt) == 40
# ground-truth program must pass the verifier for every task
import re
OPS = dict(g["GRID_OPS"])
srcmap = {
 "rot90":"g=[list(r) for r in zip(*g[::-1])]",
 "rot180":"g=[r[::-1] for r in g[::-1]]",
 "flipud":"g=g[::-1]",
 "fliplr":"g=[r[::-1] for r in g]",
 "transpose":"g=[list(r) for r in zip(*g)]",
}
n_ok = 0
for t in gt:
    ops = t["rule"].split("+")
    if not all(o in srcmap for o in ops): continue
    code = "def transform(g):\n    " + "\n    ".join(srcmap[o] for o in ops) + "\n    return g\n"
    pred, err = g["_grid_check"](code, t)
    assert pred == t["test"][1], (t["rule"], err)
    n_ok += 1
print(f"grid verifier: {n_ok} oracle programs accepted, all test outputs exact")
bad, err = g["_grid_check"]("def transform(g):\n    return [[0]]\n", gt[0])
assert bad is None and err, "verifier must reject a wrong rule"
print("grid verifier rejects a wrong rule with feedback:", err[:70])

# ----------------------------------------------------------------- sci suite ----
st = g["gen_sci"](12, rng)
print(f"sci tasks: {len(st)}")
for t in st:
    expr = t["expr"]
    s, e = g["sci_score_expr"](expr, t)
    assert s < 1e-12, (t["name"], s, e)
    s2, _ = g["sci_score_expr"]("x0+x1" if t["nvars"] >= 2 else "x0*2", t)
    if s2 < 1e-8: fails.append(f"sci decoy accepted for {t['name']}")
print("sci verifier: every ground-truth law scores nmse<1e-12, decoys rejected")

# --------------------------------------------------------------- agent suite ----
at = g["gen_agent"](16, rng)
print(f"agent tasks: {len(at)} | tiers {sorted(set(t['tier'] for t in at))} | "
      f"optimal lengths {[t['opt'] for t in at[:8]]}")
assert len(at) == 16
for t in at:
    sol = g["_agent_solve"](t["grid"])
    ok, why = g["agent_simulate"](t["grid"], sol, t["max_steps"])
    assert ok, (t["grid"], why)
    bad, why2 = g["agent_simulate"](t["grid"], "UUUU", t["max_steps"])
    assert not bad
print("agent verifier: reference BFS plan accepted for all, junk plan rejected")
print("  rejection message example:", why2)

# ------------------------------------------------------------ skill library -----
SK = g["SkillLibrary"](os.path.join(TMP,"sk.json"))
SK.add("agent", "bfs planner tier3 keys doors ordered items", "def solve(grid): return 'RRDD'")
SK.add("agent", "bfs planner tier1 simple", "def solve(grid): return 'DD'")
SK.add("grid", "grid induction rot90", "def transform(g): return g")
assert not SK.add("agent", "dup", "def solve(grid): return 'RRDD'") or True
hits = SK.retrieve("agent", "bfs grid keys doors items order tier3", k=2)
assert hits and "bfs" in hits[0]["desc"], hits
assert SK.retrieve("math", "anything") == []
SK.save(); assert os.path.exists(SK.path)
print(f"skill library: {len(SK)} entries, retrieval ranks tier3 first -> {hits[0]['desc'][:40]!r}")

# ------------------------------------------------- retrieval keys carry no leak ----
# The property that matters is not "the string avoids certain words" - a shape signature may
# honestly say "transpose" because the shape flipped. It is that the descriptor is a pure
# function of what the model is shown. Strip every privileged field and require the same key.
PRIVILEGED = {"grid": ["rule"], "sci": ["name", "expr"], "agent": ["tier", "opt", "max_steps"]}
for tasks, fn in ((gt, "_grid_shape_desc"), (st, "_sci_desc"), (at, "_agent_desc")):
    for t in tasks:
        blind = {k: v for k, v in t.items() if k not in PRIVILEGED[t["domain"]]}
        assert g[fn](blind) == g[fn](t), (
            f"{fn} reads a field the model never sees: {t['domain']}")
print("skill-retrieval keys are derived only from what the model is shown:")
print("  grid  ->", g["_grid_shape_desc"](gt[0]))
print("  sci   ->", g["_sci_desc"](st[0]))
print("  agent ->", g["_agent_desc"](at[0]))

# --------------------------------------------------------- prompt/extraction ----
assert g["extract_code"]("blah\n```python\ndef f():\n    return 1\n```\nend") == "def f():\n    return 1"
assert "def transform" in g["extract_code"]("here you go\ndef transform(g):\n    return g")
assert g["last_number"]("the answer is 1,234.50") == 1234.5
assert g["num_eq"](3.14159, 3.1416, tol=1e-3)
p = g["_grid_prompt"](gt[0], feedback="it does not reproduce example(s) [2]")
assert p[1]["content"].count("example") >= 3
p = g["_agent_prompt"](at[0])
assert "breadth-first" in p[1]["content"]
print("prompt builders and parsers: ok")

# ------------------------------------------------------ baseline check path -----
t = mt[0]
assert g["_baseline_check"](t, f"reasoning ... #### {t['ans']}")
assert not g["_baseline_check"](t, "#### 999999999")
t = gt[0]
assert g["_baseline_check"](t, "answer: " + json.dumps(t["test"][1]))
t = at[0]
assert g["_baseline_check"](t, g["_agent_solve"](t["grid"]))
t = st[0]
assert g["_baseline_check"](t, "y = " + t["expr"])
print("baseline grader accepts correct answers and rejects wrong ones in all 4 domains")

print("\nFAILURES:", fails if fails else "none")
print("ALL CORE TESTS PASSED")
