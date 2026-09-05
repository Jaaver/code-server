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
    def attach_lora(self, r=16, alpha=32): return 4_300_000
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

llm, report = G["main"]()

assert report["params"] < 1e9
for key in ("baseline","prism_v0","prism_v1","curve","grid_heldout_v0","suite_sizes"):
    assert key in report, f"missing {key}"
assert len(report["curve"]) == 3, report["curve"]
assert report["prism_v0"]["MEAN"] > report["baseline"]["MEAN"], "search must beat single pass"
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
