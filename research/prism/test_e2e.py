import os
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_testtmp"); os.makedirs(TMP, exist_ok=True)
PRISM = os.path.join(HERE, "prism_colab.py")
"""End-to-end wiring test: real solvers/verifiers/harvest/evaluate, mock LLM."""
import sys, types, os, json, random, math

def stub(name, **a):
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in a.items()]; sys.modules[name] = m; return m
np = stub("numpy"); np.random = types.SimpleNamespace(seed=lambda *a: None)
class _C:
    is_available=staticmethod(lambda: False); is_bf16_supported=staticmethod(lambda: False)
    get_device_name=staticmethod(lambda i: "stub"); empty_cache=staticmethod(lambda: None)
    OutOfMemoryError=RuntimeError
torch = stub("torch", cuda=_C, bfloat16="bf16", float16="fp16", float32="fp32",
             manual_seed=lambda *a: None, __version__="stub", no_grad=lambda: (lambda f: f))
torch.optim = types.SimpleNamespace(AdamW=None, lr_scheduler=types.SimpleNamespace(OneCycleLR=None))
torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=None))
stub("transformers", AutoTokenizer=None, AutoModelForCausalLM=None, __version__="stub")
stub("peft", LoraConfig=None, get_peft_model=None)

SCR = TMP
os.environ["PRISM_STATE"] = os.path.join(SCR, "state_e2e")
G = {"__name__": "prism_e2e"}
exec(compile(open(PRISM).read(), "prism_colab.py", "exec"), G)

# ------------------------------------------------------------------ mock LLM ----
OPSRC = {
 "rot90":"g=[list(r) for r in zip(*g[::-1])]", "rot180":"g=[r[::-1] for r in g[::-1]]",
 "flipud":"g=g[::-1]", "fliplr":"g=[r[::-1] for r in g]",
 "transpose":"g=[list(r) for r in zip(*g)]",
 "tile2x2":"g=[r+r for r in g]+[r+r for r in g]", "mirror_h":"g=[r+r[::-1] for r in g]",
 "scale2":"g=[[c for c in r for _ in (0,1)] for r in g for _ in (0,1)]",
 "shift_right":"g=[[r[-1]]+r[:-1] for r in g]",
 "colorswap":"g=[[{1:2,2:1,3:4,4:3,5:6,6:5,7:8,8:7}.get(c,c) for c in r] for r in g]",
 "border5":"h,w=len(g),len(g[0]);g=[[(5 if (r in (0,h-1) or c in (0,w-1)) else g[r][c]) for c in range(w)] for r in range(h)]",
 "gravity":("h,w=len(g),len(g[0])\n    o=[[0]*w for _ in range(h)]\n"
            "    for c in range(w):\n        col=[g[r][c] for r in range(h) if g[r][c]!=0]\n"
            "        for i,v in enumerate(col): o[h-len(col)+i][c]=v\n    g=o"),
 "denoise":("from collections import Counter\n    cnt=Counter(c for r in g for c in r if c!=0)\n"
            "    k=cnt.most_common(1)[0][0] if cnt else 0\n    g=[[(c if c==k else 0) for c in r] for r in g]"),
 "crop_bbox":("rs=[i for i,r in enumerate(g) if any(r)]\n"
              "    cs=[j for j in range(len(g[0])) if any(g[i][j] for i in range(len(g)))]\n"
              "    g=[[g[i][j] for j in range(cs[0],cs[-1]+1)] for i in range(rs[0],rs[-1]+1)] if rs and cs else g"),
}

class MockLLM:
    """Behaves like LLM.chat but answers from an oracle, with a configurable error rate
    so the refinement / voting / rejection paths are all exercised."""
    def __init__(self, tasks, p_correct=0.55, seed=3):
        self.model_id="mock"; self.n_params=596_000_000; self.calls=0; self.gen_tokens=0
        self.rng=random.Random(seed); self.p=p_correct
        self.by_q={}; self.by_grid={}; self.by_rows={}; self.by_gridworld={}
        for t in tasks: self.index(t)
    def index(self,t):
        d=t["domain"]
        if d=="math": self.by_q[t["q"][:60]]=t
        elif d=="grid": self.by_grid[json.dumps(t["train"][0][0])]=t
        elif d=="sci":
            self.by_rows["  ".join(f"{v:g}" for v in t["rows"][0])]=t   # PRISM prompt format
            self.by_rows[" ".join(str(v) for v in t["rows"][0])]=t      # baseline prompt format
        else: self.by_gridworld["\n".join(t["grid"])]=t
    def _render(self,msgs): return "".join(m["content"] for m in msgs)
    def _answer(self,prompt):
        for k,t in self.by_gridworld.items():
            if k in prompt: return self._agent(t)
        for k,t in self.by_grid.items():
            if k in prompt: return self._grid(t)
        for k,t in self.by_rows.items():
            if k in prompt: return self._sci(t)
        for k,t in self.by_q.items():
            if k in prompt: return self._math(t)
        return "I do not know."
    def _math(self,t):
        if "final numeric answer after" in "": pass
        return f"```python\nprint({t['ans']!r})\n```"
    def _grid(self,t):
        ops=t["rule"].split("+")
        if all(o in OPSRC for o in ops):
            return "```python\ndef transform(g):\n    "+"\n    ".join(OPSRC[o] for o in ops)+"\n    return g\n```"
        tbl={json.dumps(a):b for a,b in t["train"]}; tbl[json.dumps(t["test"][0])]=t["test"][1]
        return ("```python\nimport json\nT="+json.dumps(tbl)+"\ndef transform(g):\n"
                "    return T[json.dumps(g)]\n```")
    def _sci(self,t):
        n=t["nvars"]; return "```python\ndef f("+",".join(f"x{j}" for j in range(n))+"):\n    return "+t["expr"]+"\n```"
    def _agent(self,t):
        sol=G["_agent_solve"](t["grid"]); return f"```python\ndef solve(grid):\n    return {sol!r}\n```"
    def chat(self,batch,n=1,max_new_tokens=384,temperature=0.8,top_p=0.95):
        self.calls+=1; out=[]
        for msgs in batch:
            p=self._render(msgs); cands=[]
            for _ in range(n):
                if self.rng.random()<self.p: cands.append(self._answer(p))
                else: cands.append(self.rng.choice([
                    "```python\nprint(oops)\n```", "no idea", "```python\ndef transform(g):\n    return [[0]]\n```",
                    "```python\ndef solve(grid):\n    return 'UUUU'\n```", "```python\ndef f(x0):\n    return x0*3.7\n```"]))
            out.append(cands)
        return out

rng = random.Random(11)
E = G["EVAL"]
E["math"]  = G["gen_math"](8, rng)
E["grid"]  = G["gen_grid"](8, rng, difficulty=2)
E["sci"]   = G["gen_sci"](4, rng)
E["agent"] = G["gen_agent"](8, rng)
allt = E["math"]+E["grid"]+E["sci"]+E["agent"]
G["CFG"].update(k_math=6,k_grid=6,k_sci=6,k_agent=6,sci_rounds=2)

# ---- perfect oracle: PRISM must reach 100% everywhere ----
llm = MockLLM(allt, p_correct=1.0)
acc,_ = G["evaluate"](llm, "PRISM (oracle proposer)", use_prism=True)
assert acc["MEAN"] == 100.0, acc
print("PRISM with a perfect proposer -> 100.0 on every suite  ✓")

# ---- noisy oracle: verifier must still filter to high accuracy ----
llm2 = MockLLM(allt, p_correct=0.35, seed=5)
acc2,_ = G["evaluate"](llm2, "PRISM (35% noisy proposer)", use_prism=True)
base2,_ = G["evaluate"](llm2, "BASE  (35% noisy, 1 pass)", use_prism=False)
print(f"noisy proposer: PRISM MEAN {acc2['MEAN']:.1f} vs single-pass {base2['MEAN']:.1f} "
      f"(+{acc2['MEAN']-base2['MEAN']:.1f})")
assert acc2["MEAN"] > base2["MEAN"], "search+verification must beat a single pass"

# ---- always-wrong proposer: verifier must never award a point ----
class Dud(MockLLM):
    def chat(self,batch,n=1,**k): return [["```python\nprint(1)\n```"]*n for _ in batch]
dud = Dud(allt); dud.p=0
acc3,_ = G["evaluate"](dud, "PRISM (adversarial dud)", use_prism=True)
assert acc3["grid"]==0 and acc3["sci"]==0 and acc3["agent"]==0, acc3
print("adversarial dud proposer -> verifier awards 0 on grid/sci/agent (no leakage)  ✓")

# ---- curriculum + harvest + skill growth ----
G["star_finetune"] = lambda *a, **k: dict(n_pairs=len(a[1]), steps=0, loss_start=0, loss_end=0)
cur = G["build_curriculum"](16, random.Random(2), dict(grid=2))
assert len(cur) >= 12, len(cur)
assert {t["domain"] for t in cur} == {"math","grid","sci","agent"}, {t["domain"] for t in cur}
assert not ({t["name"] for t in cur if t["domain"]=="sci"} & {t["name"] for t in E["sci"]}), "eval leak!"
llm3 = MockLLM(allt+cur, p_correct=0.8, seed=9)
n0 = len(G["SKILLS"])
pairs, stats = G["harvest"](llm3, cur, k=4)
print("harvest pass-rates:", {k:f"{a}/{b}" for k,(a,b) in stats.items()},
      f"| traces={len(pairs)} | skills {n0}->{len(G['SKILLS'])}")
assert len(pairs) > 0 and len(G["SKILLS"]) > n0
assert all(isinstance(p,tuple) and len(p)==2 and "```python" in p[1] for p in pairs)
# retrieval now injects real skills into prompts
blk = G["_skill_block"]("agent","bfs keys doors ordered items tier2")
assert "```python" in blk, blk
print("skill library feeds retrieved solutions back into prompts  ✓")

# ---- persistence across "sessions" ----
G["SKILLS"].save()
S2 = G["SkillLibrary"](G["SKILLS"].path)
assert len(S2) == len(G["SKILLS"]) and len(S2) > 0
print(f"skill library persists: reloaded {len(S2)} entries from disk  ✓")
print("\nALL END-TO-END TESTS PASSED")
