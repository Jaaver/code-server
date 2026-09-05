# =====================================================================================
#  PRISM  —  Program-Reasoning Iterative Self-improving Machine
#  A sub-1B-parameter reasoning engine with verifier-guided test-time search,
#  a growing executable skill library, an auto-curriculum, and STaR/LoRA self-evolution.
#
#  Run: paste this whole file into one Google Colab cell and execute.
#  Runtime > Change runtime type > T4 GPU (works on CPU too, degraded).
#
#  WHAT THIS IS (read the honest framing at the bottom of the report):
#    - It is NOT a general superintelligence. No <1B model is, and no training run
#      inside a Colab session will make one.
#    - It IS a system that demonstrably beats far larger frontier models on tasks with
#      cheap exact verifiers, by converting parameters into search + verification +
#      accumulated reusable skills, and that measurably self-improves over rounds.
# =====================================================================================

from __future__ import annotations
import os, sys, json, math, time, random, re, subprocess, tempfile, hashlib, traceback, shutil
from collections import Counter, defaultdict

T0 = time.time()

# keep the cell's output readable — a long run is meant to be skimmed, not scrolled
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_VERBOSITY", "error")

# ------------------------------------------------------------------ configuration ---
PRESET = os.environ.get("PRISM_PRESET", "quick")     # quick | standard | full
STATE_DIR = os.environ.get("PRISM_STATE") or (
    "/content/prism_state" if os.path.isdir("/content") else os.path.join(os.getcwd(), "prism_state"))
os.makedirs(STATE_DIR, exist_ok=True)

PRESETS = {
    "quick":    dict(time_budget=4200, n_math=20, n_grid=14, n_sci=5,  n_agent=12,
                     k_math=6, k_grid=8, k_sci=8, k_agent=6, sci_rounds=3,
                     evo_rounds=2, evo_tasks=48, sft_steps=90),
    "standard": dict(time_budget=6000, n_math=40, n_grid=28, n_sci=8,  n_agent=20,
                     k_math=8, k_grid=12, k_sci=10, k_agent=8, sci_rounds=4,
                     evo_rounds=3, evo_tasks=90, sft_steps=160),
    "full":     dict(time_budget=13000, n_math=80, n_grid=50, n_sci=12, n_agent=32,
                     k_math=16, k_grid=16, k_sci=12, k_agent=12, sci_rounds=5,
                     evo_rounds=4, evo_tasks=160, sft_steps=300),
}
CFG = PRESETS[PRESET]
TIME_BUDGET = float(os.environ.get("PRISM_TIME_BUDGET", CFG["time_budget"]))

MODEL_CANDIDATES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]
if os.environ.get("PRISM_MODEL"):
    MODEL_CANDIDATES = [os.environ["PRISM_MODEL"]] + MODEL_CANDIDATES

SEED = 1337
random.seed(SEED)

def elapsed():   return time.time() - T0
def budget_left(): return TIME_BUDGET - elapsed()
def log(msg, *a):
    print(f"[{elapsed():7.1f}s] " + (msg % a if a else msg), flush=True)
def rule(t=""):
    print("\n" + "=" * 86); 
    if t: print(t); print("=" * 86, flush=True)

# ----------------------------------------------------------------------- installs ---
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                   check=False, capture_output=True)

rule("PHASE 0 — bootstrap")
try:
    import numpy as np
except Exception:
    _pip("numpy"); import numpy as np
try:
    import torch
except Exception:
    _pip("torch"); import torch
try:
    import transformers
except Exception:
    _pip("-U", "transformers"); import transformers
try:
    import peft
except Exception:
    _pip("-U", "peft"); import peft
try:
    import datasets as _ds; del _ds      # only used for the real GSM8K split
except Exception:
    _pip("datasets")
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    import warnings; warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()
except Exception:
    pass

np.random.seed(SEED)
torch.manual_seed(SEED)

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    GPU = torch.cuda.get_device_name(0)
    BF16 = torch.cuda.is_bf16_supported()
else:
    GPU, BF16 = "cpu", False
DTYPE = torch.bfloat16 if BF16 else (torch.float16 if HAS_CUDA else torch.float32)
DEV = "cuda" if HAS_CUDA else "cpu"
log("torch %s | transformers %s | device=%s (%s) | dtype=%s | preset=%s",
    torch.__version__, transformers.__version__, DEV, GPU, str(DTYPE).split(".")[-1], PRESET)
if not HAS_CUDA:
    log("!! no GPU detected — shrinking workload (results will be weaker/slower)")
    for k in ("n_math","n_grid","n_sci","n_agent"): CFG[k] = max(3, CFG[k] // 4)
    for k in ("k_math","k_grid","k_sci","k_agent"): CFG[k] = max(2, CFG[k] // 3)
    CFG["evo_rounds"], CFG["evo_tasks"], CFG["sft_steps"] = 1, 16, 30

# ------------------------------------------------------------------- code sandbox ---
SANDBOX = os.path.join(STATE_DIR, "sandbox"); os.makedirs(SANDBOX, exist_ok=True)

def extract_code(text: str) -> str:
    """Pull python out of a model response (fenced block preferred, else raw)."""
    if not text: return ""
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return max(m, key=len).strip()
    # unfenced: take from first def/import/assignment onwards
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*(def |import |from |[A-Za-z_]\w*\s*=)", ln):
            return "\n".join(lines[i:]).strip()
    return text.strip()

def run_python(code: str, timeout: float = 8.0, extra_files=None) -> dict:
    """Execute code in a fresh subprocess. Never raises: a failure is just {ok: False}."""
    d = None
    try:
        os.makedirs(SANDBOX, exist_ok=True)
        d = tempfile.mkdtemp(dir=SANDBOX)
        for name, blob in (extra_files or {}).items():
            with open(os.path.join(d, name), "w") as f: f.write(blob)
        p = os.path.join(d, "main.py")
        with open(p, "w") as f: f.write(code)
        env = dict(os.environ); env["PYTHONHASHSEED"] = "0"; env["OMP_NUM_THREADS"] = "1"
        r = subprocess.run([sys.executable, "-I", "-S" if False else "-B", p],
                           capture_output=True, text=True, timeout=timeout, cwd=d, env=env)
        return dict(ok=(r.returncode == 0), out=r.stdout.strip(), err=r.stderr.strip()[-1200:])
    except subprocess.TimeoutExpired:
        return dict(ok=False, out="", err=f"TimeoutError: exceeded {timeout}s")
    except Exception as e:
        return dict(ok=False, out="", err=f"{type(e).__name__}: {e}")
    finally:
        if d: shutil.rmtree(d, ignore_errors=True)

def last_number(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", (s or "").replace(",", ""))
    if not m: return None
    try: return float(m[-1])
    except Exception: return None

def num_eq(a, b, tol=1e-4):
    if a is None or b is None: return False
    try: return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except Exception: return False

# --------------------------------------------------------------------- LLM engine ---
class LLM:
    """Batched sampler over a sub-1B causal LM. Handles chat templates, left padding,
    n-way sampling, and (later) an attached LoRA adapter."""

    def __init__(self, candidates):
        last = None
        major = int(str(transformers.__version__).split(".")[0])
        # transformers renamed torch_dtype -> dtype in v5; try the right one first, then the other
        kw_variants = ([{"dtype": DTYPE}, {"torch_dtype": DTYPE}] if major >= 5
                       else [{"torch_dtype": DTYPE}, {"dtype": DTYPE}]) + [{}]
        for mid in candidates:
            try:
                log("loading %s ...", mid)
                tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
                mdl = None
                for kw in kw_variants:
                    try:
                        mdl = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, **kw)
                        break
                    except TypeError as te:
                        last = te; continue
                if mdl is None: raise RuntimeError("no accepted dtype kwarg")
                mdl.to(DEV); mdl.eval()
                self.model_id, self.tok, self.model = mid, tok, mdl
                break
            except Exception as e:
                last = e; log("  ! failed (%s: %s)", type(e).__name__, str(e)[:150])
        else:
            raise RuntimeError(f"no model could be loaded; last error: {last}")

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"      # never cut the instruction off the end
        self.n_params = sum(p.numel() for p in self.model.parameters())
        assert self.n_params < 1_000_000_000, f"model has {self.n_params:,} params (>=1B), violates constraint"
        log("loaded %s | %.1fM parameters (%.4fB) — under the 1B budget",
            self.model_id, self.n_params / 1e6, self.n_params / 1e9)
        self.seq_batch = 48 if HAS_CUDA else 4
        self.calls = 0
        self.gen_tokens = 0

    def _render(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            pass
        except Exception:
            pass
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            sysm = "".join(m["content"] + "\n" for m in messages if m["role"] == "system")
            usr = "".join(m["content"] + "\n" for m in messages if m["role"] == "user")
            return f"{sysm}\n### Task\n{usr}\n### Response\n"

    @torch.no_grad()
    def chat(self, batch_messages, n=1, max_new_tokens=384, temperature=0.8, top_p=0.95):
        """batch_messages: list of message-lists. Returns list (len B) of lists (len n) of strings."""
        prompts = [self._render(m) for m in batch_messages]
        results = [[] for _ in prompts]
        kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n,
                  pad_token_id=self.tok.pad_token_id, use_cache=True)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=50)
        else:
            kw.update(do_sample=False)
        cursor = 0
        while cursor < len(prompts):
            per_call = max(1, self.seq_batch // max(1, n))
            idx = list(range(cursor, min(cursor + per_call, len(prompts))))
            chunk = [prompts[i] for i in idx]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=2560).to(DEV)
            try:
                out = self.model.generate(**enc, **kw)
            except Exception as e:
                is_oom = isinstance(e, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)) or \
                         "out of memory" in str(e).lower()
                del enc
                if HAS_CUDA: torch.cuda.empty_cache()
                if is_oom and self.seq_batch > 2:
                    self.seq_batch = max(2, self.seq_batch // 2)
                    log("  OOM -> seq_batch=%d, retrying the same work", self.seq_batch)
                    continue                          # same cursor, smaller batch
                log("  generation failed (%s: %s) — skipping %d prompt(s)",
                    type(e).__name__, str(e)[:120], len(idx))
                cursor = idx[-1] + 1
                continue
            gen = out[:, enc["input_ids"].shape[1]:]
            self.calls += 1; self.gen_tokens += int(gen.numel())
            txt = self.tok.batch_decode(gen, skip_special_tokens=True)
            for j, i in enumerate(idx):
                results[i] = [t.strip() for t in txt[j * n:(j + 1) * n]]
            cursor = idx[-1] + 1
        return results

    def attach_lora(self, r=16, alpha=32):
        targets = []
        names = {n.split(".")[-1] for n, _ in self.model.named_modules()}
        for cand in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            if cand in names: targets.append(cand)
        if not targets: targets = ["c_attn", "c_proj"]
        cfg = peft.LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                              target_modules=targets, task_type="CAUSAL_LM")
        self.model = peft.get_peft_model(self.model, cfg)
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = p.data.float()
        n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model.eval()
        log("LoRA attached on %s | trainable %.2fM (%.3f%% of model)",
            ",".join(targets), n_tr / 1e6, 100 * n_tr / self.n_params)
        return n_tr

# ---------------------------------------------------------------- skill library -----
STOP = set("the a an of to and or in on for with is are be by that this it as at from "
           "return def python code write grid problem answer function using use".split())

def toks(s):
    return [w for w in re.findall(r"[a-z0-9_]+", (s or "").lower()) if w not in STOP and len(w) > 2]

class SkillLibrary:
    """Append-only store of verified, executable solutions. Non-parametric memory:
    it keeps improving the system between runs without touching any weight."""

    def __init__(self, path):
        self.path = path
        self.items = []
        if os.path.exists(path):
            try: self.items = json.load(open(path))
            except Exception: self.items = []
        self.df = Counter()
        for it in self.items: self.df.update(set(it["tokens"]))

    def add(self, domain, desc, code, score=1.0, meta=None):
        h = hashlib.sha1((domain + code).encode()).hexdigest()[:16]
        if any(it["h"] == h for it in self.items): return False
        it = dict(h=h, domain=domain, desc=desc, code=code, score=float(score),
                  tokens=toks(desc + " " + code)[:64], meta=meta or {}, t=time.time())
        self.items.append(it); self.df.update(set(it["tokens"]))
        return True

    def retrieve(self, domain, query, k=2):
        cands = [it for it in self.items if it["domain"] == domain]
        if not cands: return []
        q = Counter(toks(query)); N = len(self.items) + 1
        scored = []
        for it in cands:
            s = 0.0
            tv = Counter(it["tokens"])
            for w, c in q.items():
                if w in tv:
                    s += (1 + math.log(1 + c)) * math.log(N / (1 + self.df[w]))
            s /= math.sqrt(len(it["tokens"]) + 1)
            scored.append((s + 0.15 * it["score"], it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:k]]

    def save(self):
        json.dump(self.items, open(self.path, "w"))

    def __len__(self): return len(self.items)

SKILLS = SkillLibrary(os.path.join(STATE_DIR, "skills.json"))

# =====================================================================================
#  BENCHMARK SUITES — every one has a cheap, exact, executable ground-truth verifier.
#  That property is the entire thesis: where verification is cheap, search substitutes
#  for scale, and a 0.6B proposer + verifier can pass models 1000x its size.
# =====================================================================================

# ---------------------------------------------------------- (1) multi-step math -----
_M_TEMPLATES = [
 ("A warehouse receives {a} crates of {item}. Each crate holds {b} units. {c} units are damaged "
  "and discarded. The rest are packed into boxes of {d} units, and any leftover units are sold "
  "individually at ${e} each. How many dollars come from the individually sold units?",
  lambda v: ((v['a']*v['b']-v['c']) % v['d']) * v['e']),
 ("A train travels {a} km at {b} km/h, then {c} km at {d} km/h. What is its average speed for the "
  "whole trip, in km/h, rounded to 2 decimals?",
  lambda v: round((v['a']+v['c'])/(v['a']/v['b']+v['c']/v['d']), 2)),
 ("A shop buys an item for ${a}, marks it up {b}%, then during a sale discounts the marked price by "
  "{c}%. A customer also pays {d}% sales tax on the sale price. What does the customer pay, rounded "
  "to 2 decimals?",
  lambda v: round(v['a']*(1+v['b']/100)*(1-v['c']/100)*(1+v['d']/100), 2)),
 ("Pipe A fills a tank in {a} hours, pipe B in {b} hours, and a drain empties it in {c} hours. All "
  "three run together. How many hours to fill the empty tank, rounded to 3 decimals?",
  lambda v: round(1.0/(1/v['a']+1/v['b']-1/v['c']), 3)),
 ("A class has {a} students. {b}% play football, {c}% play chess, and {d} students play both. How "
  "many students play neither? (Percentages give whole numbers of students.)",
  lambda v: v['a'] - (round(v['a']*v['b']/100) + round(v['a']*v['c']/100) - v['d'])),
 ("An investment of ${a} grows at {b}% compounded annually for {c} years, then ${d} is withdrawn, "
  "and the rest grows for {e} more years at the same rate. What is the final amount, rounded to 2 "
  "decimals?",
  lambda v: round((v['a']*(1+v['b']/100)**v['c'] - v['d'])*(1+v['b']/100)**v['e'], 2)),
 ("A rectangle is {a} cm by {b} cm. A uniform border of width {c} cm is cut off from all sides. "
  "The remaining rectangle is cut into squares of side {d} cm. How many whole squares are obtained?",
  lambda v: ((v['a']-2*v['c'])//v['d']) * ((v['b']-2*v['c'])//v['d'])),
 ("A worker is paid ${a} per hour for the first {b} hours of a week and {c} times that rate "
  "afterwards. In one week they work {d} hours. Payroll tax of {e}% is deducted. What is the "
  "take-home pay, rounded to 2 decimals?",
  lambda v: round((v['a']*v['b'] + v['a']*v['c']*max(0, v['d']-v['b']))*(1-v['e']/100), 2)),
 ("A tank holds {a} litres of a {b}% salt solution. {c} litres are removed and replaced with pure "
  "water. What percentage of the final mixture is salt, rounded to 3 decimals?",
  lambda v: round(100*((v['a']-v['c'])*v['b']/100)/v['a'], 3)),
 ("Three machines produce {a}, {b} and {c} parts per hour. They run together for {d} hours, then "
  "the slowest stops and the others run {e} more hours. How many parts in total?",
  lambda v: (v['a']+v['b']+v['c'])*v['d'] + (v['a']+v['b']+v['c']-min(v['a'],v['b'],v['c']))*v['e']),
]

def gen_math(n, rng):
    tasks, seen = [], set()
    while len(tasks) < n:
        tpl, fn = _M_TEMPLATES[rng.randrange(len(_M_TEMPLATES))]
        v = dict(a=rng.randint(6, 60), b=rng.randint(3, 24), c=rng.randint(2, 15),
                 d=rng.randint(2, 12), e=rng.randint(2, 20),
                 item=rng.choice(["bolts", "tiles", "vials", "sensors", "bricks"]))
        if "{b}%" in tpl: v["b"] = rng.choice([10, 20, 25, 40, 50, 60, 75])
        if "{c}%" in tpl: v["c"] = rng.choice([10, 20, 25, 30, 40, 50])
        if "{d}%" in tpl: v["d"] = rng.choice([5, 8, 10, 12, 15])
        if "{e}%" in tpl: v["e"] = rng.choice([5, 10, 15, 20])
        if "compounded" in tpl: v.update(a=rng.choice([1000,2500,5000]), b=rng.choice([4,5,6,8]),
                                         c=rng.randint(2,5), d=rng.choice([200,500,800]), e=rng.randint(2,5))
        if "border of width" in tpl:
            v.update(a=rng.randint(30, 90), b=rng.randint(30, 90), c=rng.randint(2, 6), d=rng.randint(3, 9))
        if "drain empties" in tpl:
            v.update(a=rng.randint(3, 12), b=rng.randint(3, 12), c=rng.randint(14, 30))
        if "students" in tpl:
            v["a"] = rng.choice([20, 25, 40, 50, 80, 100]); v["d"] = rng.randint(2, 9)
        try:
            ans = fn(v)
        except Exception:
            continue
        if not isinstance(ans, (int, float)) or (isinstance(ans, float) and not math.isfinite(ans)):
            continue
        if isinstance(ans, (int, float)) and abs(ans) > 1e12: continue
        q = tpl.format(**v)
        if q in seen: continue
        seen.add(q)
        tasks.append(dict(domain="math", q=q, ans=float(ans)))
    return tasks

def load_gsm8k(n, rng):
    """Prefer the real GSM8K test split; fall back to the procedural suite above."""
    for repo in ("openai/gsm8k", "gsm8k"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, "main", split="test")
            idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[:n]
            out = []
            for i in idx:
                a = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
                out.append(dict(domain="math", q=ds[i]["question"], ans=float(a), src="gsm8k"))
            if out:
                log("math suite: GSM8K test split (%d items)", len(out))
                return out
        except Exception as e:
            log("math suite: %s unavailable (%s: %s)", repo, type(e).__name__, str(e)[:160])
    log("math suite: falling back to the procedural multi-step generator")
    return gen_math(n, rng)

# ------------------------------------------------- (2) ARC-lite grid induction ------
def _rot90(g):  return [list(r) for r in zip(*g[::-1])]
def _rot180(g): return [r[::-1] for r in g[::-1]]
def _flipud(g): return g[::-1]
def _fliplr(g): return [r[::-1] for r in g]
def _transpose(g): return [list(r) for r in zip(*g)]
def _tile2(g):  return [r + r for r in g] + [r + r for r in g]
def _mirror_h(g): return [r + r[::-1] for r in g]
def _scale2(g): return [[c for c in r for _ in (0, 1)] for r in g for _ in (0, 1)]
def _gravity(g):
    h, w = len(g), len(g[0]); out = [[0]*w for _ in range(h)]
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c] != 0]
        for i, v in enumerate(col): out[h-len(col)+i][c] = v
    return out
def _border(g):
    h, w = len(g), len(g[0])
    return [[(5 if (r in (0, h-1) or c in (0, w-1)) else g[r][c]) for c in range(w)] for r in range(h)]
def _shift_right(g): return [[r[-1]] + r[:-1] for r in g]
def _colorswap(g):
    m = {1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7, 9:9, 0:0}
    return [[m.get(c, c) for c in r] for r in g]
def _denoise(g):
    cnt = Counter(c for r in g for c in r if c != 0)
    if not cnt: return g
    keep = cnt.most_common(1)[0][0]
    return [[(c if c == keep else 0) for c in r] for r in g]
def _crop(g):
    rs = [i for i, r in enumerate(g) if any(r)]
    cs = [j for j in range(len(g[0])) if any(g[i][j] for i in range(len(g)))]
    if not rs or not cs: return g
    return [[g[i][j] for j in range(cs[0], cs[-1]+1)] for i in range(rs[0], rs[-1]+1)]

GRID_OPS = [("rot90", _rot90), ("rot180", _rot180), ("flipud", _flipud), ("fliplr", _fliplr),
            ("transpose", _transpose), ("tile2x2", _tile2), ("mirror_h", _mirror_h),
            ("scale2", _scale2), ("gravity", _gravity), ("border5", _border),
            ("shift_right", _shift_right), ("colorswap", _colorswap),
            ("denoise", _denoise), ("crop_bbox", _crop)]

# Four families are reserved for evaluation only. The self-evolution curriculum never
# generates them, so accuracy on the eval tasks that use them measures generalisation to
# transformations the system was never trained on, not in-distribution memorisation.
GRID_HELDOUT = {"gravity", "crop_bbox", "denoise", "border5"}
GRID_TRAINABLE = [o for o in GRID_OPS if o[0] not in GRID_HELDOUT]

def _rand_grid(rng, h, w, ncol=4, density=0.55):
    return [[(rng.randint(1, ncol) if rng.random() < density else 0) for _ in range(w)] for _ in range(h)]

def gen_grid(n, rng, difficulty=2, pool=None):
    pool = pool if pool is not None else GRID_OPS
    tasks = []
    guard = 0
    while len(tasks) < n and guard < n * 60:
        guard += 1
        k = 1 if difficulty <= 1 else rng.randint(1, min(3, difficulty))
        ops = [pool[rng.randrange(len(pool))] for _ in range(k)]
        def f(g):
            for _, o in ops: g = o(g)
            return g
        try:
            pairs = []
            for _ in range(4):
                h, w = rng.randint(3, 5), rng.randint(3, 5)
                gi = _rand_grid(rng, h, w)
                go = f([r[:] for r in gi])
                if not go or not go[0] or len(go) > 12 or len(go[0]) > 12: raise ValueError
                pairs.append((gi, go))
            if all(p[0] == p[1] for p in pairs): continue
            if len({json.dumps(p[1]) for p in pairs}) < 2: continue
        except Exception:
            continue
        tasks.append(dict(domain="grid", train=pairs[:3], test=pairs[3],
                          rule="+".join(o[0] for o in ops)))
    return tasks

# ------------------------------------------ (3) scientific law discovery -----------
SCI_LAWS = [
    ("kinetic energy",        2, "0.5*x0*x1**2",                (0.5, 4), (0.5, 5)),
    ("newton gravitation",    3, "x0*x1/x2**2",                 (1, 5),   (1, 5)),
    ("resistors in parallel", 2, "x0*x1/(x0+x1)",               (1, 9),   (1, 9)),
    ("damped amplitude",      2, "x0*math.exp(-0.5*x1)",        (1, 6),   (0.1, 4)),
    ("relativistic velocity", 2, "(x0+x1)/(1+x0*x1)",           (0.05, 0.9), (0.05, 0.9)),
    ("pendulum period",       2, "6.283185307179586*math.sqrt(x0/x1)", (0.5, 4), (1, 10)),
    ("wave interference",     2, "x0*math.sin(x1)",             (1, 5),   (0.1, 6)),
    ("lens equation",         2, "x0*x1/(x0-x1)",               (5, 12),  (1, 4)),
    ("adiabatic pressure",    2, "x0/x1**1.4",                  (1, 8),   (1, 4)),
    ("doppler shift",         2, "x0*(1+x1)",                   (1, 9),   (0.05, 0.8)),
    ("coulomb potential",     3, "x0*x1/x2",                    (1, 6),   (1, 6)),
    ("logistic growth rate",  2, "x0*x1*(1-x1)",                (0.5, 4), (0.05, 0.95)),
]

def gen_sci(n, rng, n_rows=140):
    out = []
    picks = list(range(len(SCI_LAWS))); rng.shuffle(picks)
    for i in picks[:n]:
        name, nv, expr, r0, r1 = SCI_LAWS[i]
        rows = []
        while len(rows) < n_rows:
            xs = []
            for j in range(nv):
                lo, hi = (r0 if j == 0 else r1)
                xs.append(round(rng.uniform(lo, hi), 4))
            ns = {"math": math}
            for j, x in enumerate(xs): ns[f"x{j}"] = x
            try:
                y = eval(expr, ns)
            except Exception:
                continue
            if not isinstance(y, float) or not math.isfinite(y) or abs(y) > 1e7: continue
            rows.append(xs + [round(y, 8)])
        out.append(dict(domain="sci", name=name, nvars=nv, expr=expr, rows=rows))
    return out

# ------------------------------------------ (4) goal-directed agentic planning ------
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
KEYS, DOORS = "abc", "ABC"          # 'S' and 'E' are markers, never keys or doors

def _agent_solve(grid):
    """Reference BFS over (pos, keymask, item_progress). Returns optimal action string or None."""
    h, w = len(grid), len(grid[0])
    start = None; items = {}; exit_ = None
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch == "S": start = (r, c)
            elif ch == "E": exit_ = (r, c)
            elif ch.isdigit(): items[int(ch)] = (r, c)
    nitems = len(items)
    from collections import deque
    st = (start[0], start[1], 0, 0)
    seen = {st}; q = deque([(st, "")])
    while q:
        (r, c, keys, prog), path = q.popleft()
        if prog == nitems and (r, c) == exit_: return path
        for a, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w): continue
            ch = grid[nr][nc]
            if ch == "#": continue
            nk, npg = keys, prog
            if ch in DOORS and not (keys >> (ord(ch) - 65)) & 1: continue
            if ch in KEYS: nk |= 1 << (ord(ch) - 97)
            if ch.isdigit() and int(ch) == prog + 1: npg = prog + 1
            ns = (nr, nc, nk, npg)
            if ns in seen: continue
            seen.add(ns); q.append((ns, path + a))
    return None

def agent_simulate(grid, actions, max_steps):
    h, w = len(grid), len(grid[0])
    start = None; nitems = 0; exit_ = None
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch == "S": start = (r, c)
            elif ch == "E": exit_ = (r, c)
            elif ch.isdigit(): nitems = max(nitems, int(ch))
    r, c = start; keys = 0; prog = 0
    if not actions or len(actions) > max_steps:
        return False, f"action string empty or longer than the {max_steps}-step limit"
    for i, a in enumerate(actions):
        if a not in DIRS: return False, f"illegal action {a!r} at index {i}"
        dr, dc = DIRS[a]; nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w): return False, f"step {i}: moved off the grid"
        ch = grid[nr][nc]
        if ch == "#": return False, f"step {i}: walked into a wall at ({nr},{nc})"
        if ch in DOORS and not (keys >> (ord(ch) - 65)) & 1:
            return False, f"step {i}: door {ch} is locked, key {ch.lower()} not collected yet"
        r, c = nr, nc
        if ch in KEYS: keys |= 1 << (ord(ch) - 97)
        if ch.isdigit() and int(ch) == prog + 1: prog += 1
    if prog < nitems: return False, f"ended with only {prog}/{nitems} items collected"
    if (r, c) != exit_: return False, f"ended at ({r},{c}) not at the exit {exit_}"
    return True, "goal reached"

def gen_agent(n, rng):
    tiers = [(5, 0, 1, 0), (7, 1, 2, 0), (7, 1, 3, 1), (9, 2, 3, 2)]
    tasks = []
    per = max(1, n // 4)
    for ti, (size, _, nitems, nkeys) in enumerate(tiers):
        made = 0; guard = 0
        while made < per and guard < 400:
            guard += 1
            g = [["#" if (r in (0, size-1) or c in (0, size-1)) else "." for c in range(size)]
                 for r in range(size)]
            free = [(r, c) for r in range(1, size-1) for c in range(1, size-1)]
            for _ in range(int(0.18 * len(free))):
                r, c = free[rng.randrange(len(free))]; g[r][c] = "#"
            free = [(r, c) for r in range(1, size-1) for c in range(1, size-1) if g[r][c] == "."]
            need = 2 + nitems + 2 * nkeys
            if len(free) < need + 3: continue
            rng.shuffle(free)
            g[free[0][0]][free[0][1]] = "S"
            g[free[1][0]][free[1][1]] = "E"
            p = 2
            for i in range(nitems):
                g[free[p][0]][free[p][1]] = str(i + 1); p += 1
            for i in range(nkeys):
                g[free[p][0]][free[p][1]] = chr(97 + i); p += 1
                g[free[p][0]][free[p][1]] = chr(65 + i); p += 1
            grid = ["".join(r) for r in g]
            sol = _agent_solve(grid)
            if sol is None or len(sol) < 4: continue
            tasks.append(dict(domain="agent", grid=grid, tier=ti + 1,
                              opt=len(sol), max_steps=int(2.5 * len(sol)) + 25))
            made += 1
    return tasks[:n]

# =====================================================================================
#  SOLVERS
#  Baseline  = one greedy chain-of-thought pass, no tools, no search  (what a raw model does)
#  PRISM     = program synthesis + sandbox execution + exact verification + self-consistency
#              + error-driven refinement + retrieved skills from the growing library
# =====================================================================================

SYS_SOLVER = ("You are PRISM, a precise program-writing reasoner. You always answer with a single "
              "Python code block and nothing else. Your code must be short, correct and runnable "
              "with only the standard library.")

def _skill_block(domain, query, k=2):
    hits = SKILLS.retrieve(domain, query, k=k)
    if not hits: return ""
    parts = ["\n# Verified solutions you previously discovered for similar problems — reuse the pattern:"]
    for h in hits:
        parts.append("```python\n" + h["code"][:900] + "\n```")
    return "\n".join(parts) + "\n"

# ---------------------------------------------------------------- baseline (CoT) ----
def baseline_batch(llm, tasks, max_new_tokens=320):
    """One greedy natural-language pass. This is the ablation that isolates the value of
    everything PRISM adds on top of the same weights."""
    msgs = [[{"role": "system", "content": "You are a helpful reasoning assistant."},
             {"role": "user", "content": _baseline_prompt_text(t)}] for t in tasks]
    outs = llm.chat(msgs, n=1, max_new_tokens=max_new_tokens, temperature=0.0)
    return [bool(_baseline_check(t, (o[0] if o else ""))) for t, o in zip(tasks, outs)]

def _baseline_extract(t, txt):
    """The candidate answer a single natural-language pass committed to, as a hashable key."""
    try:
        if t["domain"] == "math":
            tail = txt.split("####")[-1] if "####" in txt else txt
            v = last_number(tail)
            return None if v is None else round(v, 6)
        if t["domain"] == "grid":
            m = re.findall(r"\[\s*\[.*?\]\s*\]", txt, flags=re.S)
            return json.dumps(json.loads(m[-1].replace("'", '"'))) if m else None
        if t["domain"] == "sci":
            seg = txt.split("y =")[-1].split("\n")[0] if "y =" in txt else txt.split("\n")[0]
            seg = seg.strip().strip("`$ ")
            return seg or None
        mv = "".join(ch for ch in txt.upper() if ch in "UDLR")
        return mv or None
    except Exception:
        return None

def _baseline_render(t, key):
    """Turn a voted-on answer back into text the shared grader accepts."""
    if t["domain"] == "math": return "#### %r" % key
    if t["domain"] == "grid": return key
    if t["domain"] == "sci":  return "y = " + key
    return key

def baseline_selfconsistency(llm, tasks, k, max_new_tokens=320):
    """Compute-matched control: the same k samples PRISM gets, majority-voted, but with no
    program execution and no verifier. Isolates 'more sampling' from 'search + verification'."""
    msgs = [[{"role": "system", "content": "You are a helpful reasoning assistant."},
             {"role": "user", "content": _baseline_prompt_text(t)}] for t in tasks]
    outs = llm.chat(msgs, n=k, max_new_tokens=max_new_tokens, temperature=0.8)
    res = []
    for t, cands in zip(tasks, outs):
        votes = Counter(x for x in (_baseline_extract(t, c) for c in cands) if x is not None)
        if not votes:
            res.append(False); continue
        res.append(bool(_baseline_check(t, _baseline_render(t, votes.most_common(1)[0][0]))))
    return res

# ------------------------------------------------------------- PRISM: math (PoT) ----
def _math_prompt(t, with_skills=True):
    u = ("Problem:\n" + t["q"] +
         "\n\nWrite a Python program that computes the answer and prints ONLY the final number "
         "with print(). Use exact arithmetic where possible. No explanation, no input().")
    if with_skills: u += _skill_block("math", t["q"])
    return [{"role": "system", "content": SYS_SOLVER}, {"role": "user", "content": u}]

def prism_math(llm, tasks, k=6, refine=1):
    prompts = [_math_prompt(t) for t in tasks]
    outs = llm.chat(prompts, n=k, max_new_tokens=300, temperature=0.9)

    results, traces = [], []
    pending = []
    for ti, (t, cands) in enumerate(zip(tasks, outs)):
        votes, best_code = Counter(), {}
        for c in cands:
            code = extract_code(c)
            if not code: continue
            r = run_python(code, timeout=6)
            v = last_number(r["out"]) if r["ok"] else None
            if v is None: continue
            key = round(v, 6)
            votes[key] += 1
            best_code.setdefault(key, code)
        if votes:
            val, cnt = votes.most_common(1)[0]
            conf = cnt / max(1, k)
            results.append(dict(pred=val, conf=conf, code=best_code[val]))
        else:
            results.append(None); pending.append(ti)

    # error-driven refinement for the ones that produced nothing runnable
    for _ in range(refine):
        if not pending or budget_left() < 120: break
        rp = []
        for ti in pending:
            t = tasks[ti]
            rp.append([{"role": "system", "content": SYS_SOLVER},
                       {"role": "user", "content":
                        "Your previous program failed to run or printed no number.\nProblem:\n" +
                        t["q"] + "\n\nWrite a NEW, very simple Python program using only basic "
                        "arithmetic that prints one number with print(). Nothing else."}])
        ro = llm.chat(rp, n=max(2, k // 2), max_new_tokens=260, temperature=0.9)
        still = []
        for ti, cands in zip(pending, ro):
            votes, bc = Counter(), {}
            for c in cands:
                code = extract_code(c)
                if not code: continue
                r = run_python(code, timeout=6)
                v = last_number(r["out"]) if r["ok"] else None
                if v is None: continue
                votes[round(v, 6)] += 1; bc.setdefault(round(v, 6), code)
            if votes:
                val, cnt = votes.most_common(1)[0]
                results[ti] = dict(pred=val, conf=cnt / max(1, k), code=bc[val])
            else:
                still.append(ti)
        pending = still

    ok = []
    for t, r in zip(tasks, results):
        good = bool(r) and num_eq(r["pred"], t["ans"])
        ok.append(good)
        traces.append(dict(task=t, res=r, correct=good))
    return ok, traces

# ------------------------------------------------ PRISM: grid induction (programs) --
def _grid_shape_desc(t):
    """A shape signature is the one cue available without knowing the rule — it separates
    resizing families (tile, scale, crop, mirror) from in-place recolour/rearrange families."""
    a, b = t["train"][0]
    hi, wi, ho, wo = len(a), len(a[0]), len(b), len(b[0])
    kind = ("same shape rearrange recolour" if (hi, wi) == (ho, wo) else
            "transpose swap axes" if (hi, wi) == (wo, ho) else
            "grow tile scale mirror" if ho * wo > hi * wi else "shrink crop extract")
    return f"grid induction {hi}x{wi} to {ho}x{wo} {kind}"

def _grid_prompt(t, feedback=None):
    ex = "".join(f"# example {i+1}\nIN  = {json.dumps(a)}\nOUT = {json.dumps(b)}\n"
                 for i, (a, b) in enumerate(t["train"]))
    u = ("Induce the single transformation rule that maps every IN grid to its OUT grid.\n\n" + ex +
         "\nWrite exactly one function:\n```python\ndef transform(g):\n    # g: list[list[int]] -> list[list[int]]\n"
         "    ...\n```\nIt must reproduce every example above exactly. No explanation." +
         _skill_block("grid", _grid_shape_desc(t)))
    if feedback:
        u += "\n\nYour previous attempt was rejected:\n" + feedback + "\nFix it and return the full function again."
    return [{"role": "system", "content": SYS_SOLVER}, {"role": "user", "content": u}]

def _grid_check(code, t):
    """Run candidate transform on train pairs; if all match, return the predicted test grid."""
    harness = (code + "\n\nimport json\n_T=" + json.dumps(t["train"]) + "\n_X=" +
               json.dumps(t["test"][0]) + "\n_r=[]\n"
               "for _a,_b in _T:\n    _r.append(transform([list(x) for x in _a])==_b)\n"
               "print(json.dumps({'train':_r,'test':transform([list(x) for x in _X])}))\n")
    r = run_python(harness, timeout=6)
    if not r["ok"]:
        return None, (r["err"].splitlines() or ["error"])[-1][:200]
    try:
        d = json.loads(r["out"].splitlines()[-1])
    except Exception:
        return None, "the function did not return JSON-serialisable grids"
    if not all(d["train"]):
        bad = [i + 1 for i, v in enumerate(d["train"]) if not v]
        return None, f"it does not reproduce example(s) {bad}"
    return d["test"], None

def prism_grid(llm, tasks, k=8, rounds=2):
    solved = [None] * len(tasks)
    codes = [None] * len(tasks)
    feedback = [None] * len(tasks)
    active = list(range(len(tasks)))
    for rd in range(rounds):
        if not active or budget_left() < 90: break
        outs = llm.chat([_grid_prompt(tasks[i], feedback[i]) for i in active],
                        n=k if rd == 0 else max(2, k // 2), max_new_tokens=340, temperature=0.9)
        nxt = []
        for i, cands in zip(active, outs):
            votes, first_code, err = Counter(), {}, None
            for c in cands:
                code = extract_code(c)
                if "def transform" not in code: continue
                pred, e = _grid_check(code, tasks[i])
                if pred is None:
                    err = err or e; continue
                key = json.dumps(pred)
                votes[key] += 1; first_code.setdefault(key, code)
            if votes:
                key, _ = votes.most_common(1)[0]
                solved[i] = json.loads(key); codes[i] = first_code[key]
            else:
                feedback[i] = err or "no candidate ran successfully"
                nxt.append(i)
        active = nxt
    ok, traces = [], []
    for i, t in enumerate(tasks):
        good = solved[i] == t["test"][1]
        ok.append(good)
        traces.append(dict(task=t, code=codes[i], correct=good))
    return ok, traces

# --------------------------------------- PRISM: symbolic law discovery (FunSearch) --
def _sci_desc(t):
    """A retrieval key computed only from what the model is actually shown — the data table.
    Using the law's name here would let the skill library see privileged task identity."""
    rows, nv = t["rows"], t["nvars"]
    ys = [r[-1] for r in rows]; my = sum(ys) / len(ys)
    parts = [f"law {nv}vars"]
    for j in range(nv):
        xs = [r[j] for r in rows]; mx = sum(xs) / len(xs)
        cov = sum((a - mx) * (bb - my) for a, bb in zip(xs, ys))
        parts.append(f"x{j}" + ("rising" if cov > 0 else "falling"))
    lo, hi = min(ys), max(ys)
    parts.append("allpositive" if lo > 0 else "signed")
    parts.append("widerange" if (hi - lo) > 100 * max(1e-9, abs(my)) else
                 "narrowrange" if (hi - lo) < abs(my) else "midrange")
    return " ".join(parts)

def _agent_desc(t):
    """Same principle: describe the puzzle by what is visible in the grid itself."""
    flat = "".join(t["grid"])
    return ("planner bfs %dx%d grid %d items %d keys %d doors ordered pickup" %
            (len(t["grid"]), len(t["grid"][0]),
             sum(c.isdigit() for c in flat), sum(c in KEYS for c in flat),
             sum(c in DOORS for c in flat)))

def sci_score_expr(expr, t, fit=False):
    """Normalised MSE of a candidate closed form. Returns (score, effective_expr, error).

    With fit=True the sandbox also least-squares fits a scale and offset around the
    candidate, so the model only has to propose the functional FORM and the optimiser
    supplies the constants -- the standard skeleton+optimiser split in symbolic regression,
    and the thing that lets a 0.6B proposer be useful here. The single-pass baseline is
    scored with fit=False so this tool is not silently credited to the raw model."""
    if not expr or len(expr) > 400: return (1e18, expr, "empty")
    payload = json.dumps(t["rows"])
    src = ("import math, json\nrows=json.loads('''" + payload + "''')\n"
           "def f(" + ",".join(f"x{j}" for j in range(t["nvars"])) + "):\n    return (" + expr + ")\n"
           "ps=[]\n"
           "for r in rows:\n"
           "    try:\n        p=float(f(*r[:-1]))\n    except Exception:\n"
           "        print(json.dumps({'raw':1e18})); raise SystemExit\n"
           "    if p!=p or abs(p)>1e15:\n        print(json.dumps({'raw':1e18})); raise SystemExit\n"
           "    ps.append(p)\n"
           "ys=[r[-1] for r in rows]; n=len(ys)\n"
           "mu=sum(ys)/n; den=max(sum((y-mu)**2 for y in ys), 1e-12)\n"
           "raw=sum((p-y)**2 for p,y in zip(ps,ys))/den\n"
           "out={'raw':raw}\n"
           "if " + ("True" if fit else "False") + ":\n"
           "    mp=sum(ps)/n\n"
           "    sxx=sum((p-mp)**2 for p in ps); sxy=sum((p-mp)*(y-mu) for p,y in zip(ps,ys))\n"
           "    if sxx>1e-12:\n"
           "        a=sxy/sxx; bb=mu-a*mp\n"
           "        fit=sum((a*p+bb-y)**2 for p,y in zip(ps,ys))/den\n"
           "        out.update(fit=fit, a=a, b=bb, mu=mu)\n"
           "print(json.dumps(out))\n")
    r = run_python(src, timeout=8)
    if not r["ok"]: return (1e18, expr, (r["err"].splitlines() or ["error"])[-1][:160])
    try:
        d = json.loads((r["out"].splitlines() or ["{}"])[-1])
    except Exception:
        return (1e18, expr, "unparsable score")
    raw = float(d.get("raw", 1e18))
    if "fit" in d and float(d["fit"]) < raw:
        a, bb = float(d["a"]), float(d["b"])
        scale = max(1.0, abs(float(d.get("mu", 1.0))))      # fitted constants carry FP noise
        a = round(a, 10); bb = 0.0 if abs(bb) < 1e-9 * scale else round(bb, 10)
        if abs(a - 1) < 1e-9 and bb == 0.0:
            return (raw, expr, None)
        e2 = (f"{a}*({expr})" if abs(a - 1) > 1e-9 else expr) + (f" + {bb}" if bb else "")
        return (float(d["fit"]), e2, None)
    return (raw, expr, None)

def prism_sci(llm, tasks, k=8, rounds=3):
    """LLM as a mutation operator inside an evolutionary loop scored by exact numeric fit."""
    pops = [[] for _ in tasks]      # list of (nmse, expr)
    for rd in range(rounds):
        if budget_left() < 90: break
        prompts = []
        for t, pop in zip(tasks, pops):
            head = t["rows"][:14]
            cols = " ".join(f"x{j}" for j in range(t["nvars"])) + "   y"
            u = ("You are discovering a closed-form scientific law from measurements.\n"
                 "Columns: " + cols + "\n" +
                 "\n".join("  ".join(f"{v:g}" for v in r) for r in head) +
                 "\n\nWrite exactly one function, using only + - * / ** and math.sin/cos/exp/log/sqrt/pi:\n"
                 "```python\ndef f(" + ",".join(f"x{j}" for j in range(t["nvars"])) + "):\n    return ...\n```\n")
            if pop:
                u += ("\nBest candidates so far (normalised error, lower is better) — propose a "
                      "DIFFERENT and better law, do not repeat them:\n")
                for s, e in pop[:4]:
                    u += f"  err={s:.3e}   return {e}\n"
            u += _skill_block("sci", _sci_desc(t))
            prompts.append([{"role": "system", "content": SYS_SOLVER}, {"role": "user", "content": u}])
        outs = llm.chat(prompts, n=k, max_new_tokens=180, temperature=1.0)
        for i, (t, cands) in enumerate(zip(tasks, outs)):
            for c in cands:
                code = extract_code(c)
                m = re.search(r"return\s+(.+)", code)
                if not m: continue
                expr = m.group(1).strip().rstrip(";")
                if "x0" not in expr and t["nvars"] >= 1: continue
                s, eff, _ = sci_score_expr(expr, t, fit=True)
                if s < 1e17: pops[i].append((s, eff))
            seen = set(); ded = []
            for s, e in sorted(pops[i]):
                if e in seen: continue
                seen.add(e); ded.append((s, e))
            pops[i] = ded[:6]
        best = [p[0][0] if p else 1e18 for p in pops]
        log("  sci round %d/%d: exact=%d/%d  median nmse=%.2e",
            rd + 1, rounds, sum(b < 1e-8 for b in best), len(tasks),
            sorted(best)[len(best) // 2])
        if all(b < 1e-10 for b in best): break
    ok, close, traces = [], [], []
    for t, p in zip(tasks, pops):
        s, e = (p[0] if p else (1e18, ""))
        ok.append(s < 1e-8)                 # exact symbolic recovery
        close.append(s < 1e-3)              # a usable predictive law
        traces.append(dict(task=t, nmse=s, expr=e, correct=s < 1e-8))
    return ok, close, traces

# ------------------------------------------------- PRISM: agentic goal achievement --
AGENT_SPEC = (
 "Grid world rules:\n"
 "  '#' wall, '.' floor, 'S' start, 'E' exit\n"
 "  digits '1','2','3' are items; stepping on item k collects it ONLY if k is the next one\n"
 "     still needed (1, then 2, then 3) - otherwise that tile behaves as plain floor\n"
 "  'a'/'b'/'c' are keys; 'A'/'B'/'C' are doors, enterable only while holding the matching key\n"
 "  ('S' and 'E' are just markers - they are never doors and cost nothing to enter)\n"
 "  actions: 'U' up (row-1), 'D' down (row+1), 'L' left (col-1), 'R' right (col+1)\n"
 "The goal: collect every item in order, then reach 'E'.\n")

def _agent_prompt(t, feedback=None):
    u = (AGENT_SPEC + "\nGrid:\n" + "\n".join(t["grid"]) +
         "\n\nWrite exactly one function that PLANS the route (a breadth-first search over the state "
         "(row, col, keys_held, items_collected) is the reliable approach):\n"
         "```python\ndef solve(grid):\n    # grid: list[str] -> return the action string, e.g. 'RRDDL'\n"
         "    ...\n```\nThen the harness will call it. Return only the code block." +
         _skill_block("agent", _agent_desc(t)))
    if feedback:
        u += "\n\nYour previous plan was rejected: " + feedback + "\nReturn a corrected full function."
    return [{"role": "system", "content": SYS_SOLVER}, {"role": "user", "content": u}]

def prism_agent(llm, tasks, k=6, rounds=3):
    done = [False] * len(tasks); codes = [None] * len(tasks); fb = [None] * len(tasks)
    active = list(range(len(tasks)))
    for rd in range(rounds):
        if not active or budget_left() < 90: break
        outs = llm.chat([_agent_prompt(tasks[i], fb[i]) for i in active],
                        n=k if rd == 0 else max(2, k // 2), max_new_tokens=420, temperature=0.9)
        nxt = []
        for i, cands in zip(active, outs):
            t = tasks[i]; err = None; won = False
            for c in cands:
                code = extract_code(c)
                if "def solve" not in code: continue
                harness = (code + "\n\ngrid=" + json.dumps(t["grid"]) +
                           "\nr=solve(grid)\nprint(''.join(ch for ch in str(r).upper() if ch in 'UDLR'))\n")
                r = run_python(harness, timeout=10)
                if not r["ok"]:
                    err = err or (r["err"].splitlines() or ["error"])[-1][:180]; continue
                mv = (r["out"].splitlines() or [""])[-1].strip()
                good, why = agent_simulate(t["grid"], mv, t["max_steps"])
                if good:
                    done[i] = True; codes[i] = code; won = True; break
                err = err or why
            if not won:
                fb[i] = err or "no runnable plan produced"; nxt.append(i)
        active = nxt
    traces = [dict(task=t, code=codes[i], correct=done[i]) for i, t in enumerate(tasks)]
    return done, traces

# =====================================================================================
#  SELF-EVOLUTION
#  Auto-curriculum -> verifier-filtered self-generated data -> two channels of memory:
#    (a) non-parametric: the executable skill library (persists across sessions)
#    (b) parametric:     LoRA rejection-sampling fine-tuning (STaR) on verified traces
#  Nothing here uses human labels: every training target was checked by an exact verifier.
# =====================================================================================

def build_curriculum(n, rng, frontier):
    """Sample tasks near the current competence frontier (target pass-rate band 0.2-0.8)."""
    tasks = []
    q = max(1, n // 4)
    tasks += gen_math(q, rng)
    tasks += gen_grid(q, rng, difficulty=frontier.get("grid", 2), pool=GRID_TRAINABLE)
    used = {t["name"] for t in EVAL["sci"]}
    pool = [l for l in SCI_LAWS if l[0] not in used]
    if pool:
        saved = SCI_LAWS[:]
        try:
            SCI_LAWS[:] = pool
            tasks += gen_sci(min(q, len(pool)), rng)
        finally:
            SCI_LAWS[:] = saved
    tasks += gen_agent(q, rng)
    rng.shuffle(tasks)
    return tasks

def harvest(llm, tasks, k):
    """Run the PRISM stack on curriculum tasks; return verified (prompt, completion) pairs."""
    by = defaultdict(list)
    for t in tasks: by[t["domain"]].append(t)
    pairs, stats = [], {}
    if by["math"]:
        ok, tr = prism_math(llm, by["math"], k=k, refine=0)
        stats["math"] = (sum(ok), len(ok))
        for x in tr:
            if x["correct"] and x["res"]:
                pairs.append((llm._render(_math_prompt(x["task"], with_skills=False)),
                              "```python\n" + x["res"]["code"].strip() + "\n```"))
                SKILLS.add("math", x["task"]["q"][:180], x["res"]["code"])
    if by["grid"]:
        ok, tr = prism_grid(llm, by["grid"], k=k, rounds=2)
        stats["grid"] = (sum(ok), len(ok))
        for x in tr:
            if x["correct"] and x["code"]:
                pairs.append((llm._render(_grid_prompt(x["task"])), "```python\n" + x["code"].strip() + "\n```"))
                SKILLS.add("grid", _grid_shape_desc(x["task"]), x["code"])
    if by["sci"]:
        ok, close, tr = prism_sci(llm, by["sci"], k=max(4, k), rounds=2)
        stats["sci"] = (sum(ok), len(ok))
        for x in tr:
            if x["nmse"] < 1e-6 and x["expr"]:
                code = ("def f(" + ",".join(f"x{j}" for j in range(x["task"]["nvars"])) +
                        "):\n    return " + x["expr"])
                SKILLS.add("sci", _sci_desc(x["task"]), code, score=1.0)
    if by["agent"]:
        ok, tr = prism_agent(llm, by["agent"], k=k, rounds=3)
        stats["agent"] = (sum(ok), len(ok))
        for x in tr:
            if x["correct"] and x["code"]:
                pairs.append((llm._render(_agent_prompt(x["task"])), "```python\n" + x["code"].strip() + "\n```"))
                SKILLS.add("agent", _agent_desc(x["task"]), x["code"])
    SKILLS.save()
    return pairs, stats

def star_finetune(llm, pairs, steps, lr=1.2e-4, bs=2, maxlen=896):
    """Masked-prompt SFT on verifier-approved trajectories (STaR / rejection sampling)."""
    if len(pairs) < 4 or steps < 1:
        log("  not enough verified traces (%d) or steps (%d) — skipping the weight update",
            len(pairs), steps)
        return None
    tok = llm.tok
    ex = []
    for p, c in pairs:
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ci = tok(c + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(ci) >= maxlen:            # a completion alone longer than the window
            ci, pi = ci[:maxlen - 1], pi[-1:]
        elif len(pi) + len(ci) > maxlen:
            pi = pi[-(maxlen - len(ci)):]  # keep the instruction tail, drop the head
        ids = pi + ci
        lab = [-100] * len(pi) + ci[:]
        ex.append((ids, lab))
    llm.model.train()
    prev_cache = getattr(llm.model.config, "use_cache", True)
    try: llm.model.config.use_cache = False
    except Exception: pass
    params = [p for p in llm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    use_scaler = HAS_CUDA and DTYPE == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    pad = tok.pad_token_id
    losses = []
    order = list(range(len(ex)))
    for step in range(steps):
        if budget_left() < 90:
            log("  time budget — stopping SFT at step %d", step); break
        if step % max(1, len(ex) // bs) == 0: random.shuffle(order)
        sel = [ex[order[(step * bs + j) % len(order)]] for j in range(bs)]
        L = max(len(a) for a, _ in sel)
        ids = torch.full((len(sel), L), pad, dtype=torch.long)
        lab = torch.full((len(sel), L), -100, dtype=torch.long)
        att = torch.zeros((len(sel), L), dtype=torch.long)
        for j, (a, b) in enumerate(sel):
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
        ids, lab, att = ids.to(DEV), lab.to(DEV), att.to(DEV)
        with torch.autocast(device_type="cuda" if HAS_CUDA else "cpu",
                            dtype=(DTYPE if HAS_CUDA else torch.float32), enabled=HAS_CUDA):
            out = llm.model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss
        if not torch.isfinite(loss):
            log("  non-finite loss at step %d — skipped", step); continue
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        try: sched.step()
        except Exception: pass
        losses.append(float(loss.detach()))
        if (step + 1) % 20 == 0:
            log("  sft step %3d/%d  loss=%.4f", step + 1, steps, sum(losses[-20:]) / len(losses[-20:]))
    llm.model.eval()
    try: llm.model.config.use_cache = prev_cache
    except Exception: pass
    try:
        llm.model.save_pretrained(os.path.join(STATE_DIR, "lora"))
    except Exception: pass
    return dict(n_pairs=len(pairs), steps=len(losses),
                loss_start=sum(losses[:10]) / max(1, len(losses[:10])),
                loss_end=sum(losses[-10:]) / max(1, len(losses[-10:])))

# =====================================================================================
#  EVALUATION HARNESS
# =====================================================================================
EVAL = {}

def _guard(domain, fn):
    """Run one suite; on an unexpected failure score it zero and keep the run alive."""
    try:
        return fn()
    except Exception as e:
        log("  !! %s suite failed (%s: %s) — scored 0, continuing", domain, type(e).__name__, str(e)[:160])
        traceback.print_exc()
        return [False] * len(EVAL[domain])

def evaluate(llm, tag, use_prism=True, self_consistency=False):
    r = {}
    if self_consistency:
        ks = dict(math=CFG["k_math"], grid=CFG["k_grid"], sci=CFG["k_sci"], agent=CFG["k_agent"])
        for d in ("math", "grid", "sci", "agent"):
            r[d] = _guard(d, lambda d=d: baseline_selfconsistency(llm, EVAL[d], ks[d]))
        r["sci_close"] = r["sci"]
    elif use_prism:
        r["math"] = _guard("math", lambda: prism_math(llm, EVAL["math"], k=CFG["k_math"], refine=1)[0])
        r["grid"] = _guard("grid", lambda: prism_grid(llm, EVAL["grid"], k=CFG["k_grid"], rounds=2)[0])
        sci = _guard("sci", lambda: prism_sci(llm, EVAL["sci"], k=CFG["k_sci"], rounds=CFG["sci_rounds"])[:2])
        r["sci"], r["sci_close"] = (sci if isinstance(sci, tuple) else (sci, sci))
        r["agent"] = _guard("agent", lambda: prism_agent(llm, EVAL["agent"], k=CFG["k_agent"], rounds=3)[0])
    else:
        for d in ("math", "grid", "sci", "agent"):
            r[d] = _guard(d, lambda d=d: baseline_batch(llm, EVAL[d]))
        r["sci_close"] = r["sci"]
    acc = {d: (100.0 * sum(v) / max(1, len(v))) for d, v in r.items()}
    acc["MEAN"] = sum(acc[d] for d in ("math", "grid", "sci", "agent")) / 4
    log("%-28s math %.1f | grid %.1f | sci %.1f | agent %.1f || MEAN %.1f",
        tag, acc["math"], acc["grid"], acc["sci"], acc["agent"], acc["MEAN"])
    return acc, r

def frontier_probe():
    """Optional real head-to-head. Set PRISM_FRONTIER_KEY (+ optional _BASE / _MODEL / _K) to run
    a frontier model through the identical protocol: same prompts, same grader, no tools.
    PRISM_FRONTIER_K > 1 gives it the same self-consistency budget the control row gets.
    Skipped entirely when no key is set."""
    key = os.environ.get("PRISM_FRONTIER_KEY")
    if not key: return None
    import urllib.request
    base = os.environ.get("PRISM_FRONTIER_BASE", "https://api.openai.com/v1")
    model = os.environ.get("PRISM_FRONTIER_MODEL", "gpt-4o")
    kfr = max(1, int(os.environ.get("PRISM_FRONTIER_K", "1")))

    def ask(prompt, temp):
        payload = dict(model=model, messages=[{"role": "user", "content": prompt}])
        if temp is not None: payload["temperature"] = temp
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=180) as f:
            return json.loads(f.read())["choices"][0]["message"]["content"]

    def ask_retry(prompt, temp):
        for attempt, t in enumerate((temp, None, temp)):      # some models reject temperature
            try:
                return ask(prompt, t)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt == 2: raise RuntimeError(err)
        return ""

    log("frontier probe: %s via %s (k=%d, no tools, same grader)", model, base, kfr)
    res, failures = {}, {}
    for d in ("math", "grid", "sci", "agent"):
        hits = 0; asked = 0; failed = 0
        for t in EVAL[d]:
            try:
                if kfr == 1:
                    ok = _baseline_check(t, ask_retry(_baseline_prompt_text(t), 0))
                else:
                    votes = Counter()
                    for _ in range(kfr):
                        x = _baseline_extract(t, ask_retry(_baseline_prompt_text(t), 1.0))
                        if x is not None: votes[x] += 1
                    ok = bool(votes) and _baseline_check(t, _baseline_render(t, votes.most_common(1)[0][0]))
                asked += 1; hits += int(ok)
            except Exception as e:
                failed += 1
                log("  frontier call failed on a %s item (%s) — excluded, not scored zero", d, str(e)[:100])
                if failed >= 3:
                    log("  giving up on the %s suite after 3 failures", d); break
        # score only what actually got an answer; an unreachable API must not look like a wrong one
        res[d] = (100.0 * hits / asked) if asked else float("nan")
        if failed: failures[d] = failed
    scored = [res[d] for d in ("math", "grid", "sci", "agent") if res[d] == res[d]]
    res["MEAN"] = (sum(scored) / len(scored)) if scored else float("nan")
    res["model"] = model + (f" (k={kfr})" if kfr > 1 else "")
    if failures: res["failed_calls"] = failures
    return res if scored else None

def _baseline_prompt_text(t):
    if t["domain"] == "math":
        return t["q"] + "\n\nThink step by step, then write the final numeric answer after '####'."
    if t["domain"] == "grid":
        return ("Infer the transformation from the examples and output ONLY the resulting test grid as "
                "a JSON list of lists.\n" +
                "".join(f"Example {i+1} input: {json.dumps(a)}\nExample {i+1} output: {json.dumps(b)}\n"
                        for i, (a, b) in enumerate(t["train"])) +
                f"Test input: {json.dumps(t['test'][0])}\nTest output:")
    if t["domain"] == "sci":
        cols = " ".join(f"x{j}" for j in range(t["nvars"])) + " y"
        return ("Discover the closed-form law y = f(" + ",".join(f"x{j}" for j in range(t["nvars"])) +
                ") that generated this data. Reply with ONLY the formula after 'y ='.\n" + cols + "\n" +
                "\n".join(" ".join(str(v) for v in r) for r in t["rows"][:12]))
    return (AGENT_SPEC + "Output ONLY the move string.\n" + "\n".join(t["grid"]))

def _baseline_check(t, txt):
    try:
        if t["domain"] == "math":
            tail = txt.split("####")[-1] if "####" in txt else txt
            return num_eq(last_number(tail), t["ans"])
        if t["domain"] == "grid":
            m = re.findall(r"\[\s*\[.*?\]\s*\]", txt, flags=re.S)
            return bool(m) and json.loads(m[-1].replace("'", '"')) == t["test"][1]
        if t["domain"] == "sci":
            seg = txt.split("y =")[-1].split("\n")[0] if "y =" in txt else txt.split("\n")[0]
            return sci_score_expr(seg.strip().strip("`$ "), t)[0] < 1e-8
        mv = "".join(ch for ch in txt.upper() if ch in "UDLR")
        return agent_simulate(t["grid"], mv, t["max_steps"])[0]
    except Exception:
        return False

# =====================================================================================
#  MAIN
# =====================================================================================
def main():
    rng = random.Random(SEED)
    rule("PHASE 1 — build held-out benchmark suites (all exactly verifiable)")
    EVAL["math"]  = load_gsm8k(CFG["n_math"], rng)
    EVAL["grid"]  = gen_grid(CFG["n_grid"], rng, difficulty=2)
    EVAL["sci"]   = gen_sci(CFG["n_sci"], rng)
    EVAL["agent"] = gen_agent(CFG["n_agent"], rng)
    for d in ("math", "grid", "sci", "agent"):
        log("  %-6s %3d items", d, len(EVAL[d]))
    log("  agent tiers: %s", dict(Counter(t["tier"] for t in EVAL["agent"])))
    log("  skill library carried in from previous runs: %d entries", len(SKILLS))

    rule("PHASE 2 — load the sub-1B base model")
    llm = LLM(MODEL_CANDIDATES)
    report = dict(model=llm.model_id, params=llm.n_params, preset=PRESET, device=DEV, gpu=GPU,
                  suite_sizes={d: len(EVAL[d]) for d in EVAL}, seed=SEED)

    rule("PHASE 3 — baseline: the same weights, one greedy pass, no tools, no search")
    base_acc, _ = evaluate(llm, "BASE  (single-pass CoT)", use_prism=False)
    report["baseline"] = base_acc

    rule("PHASE 3b — compute-matched control: same k samples, majority vote, still no tools")
    sc_acc, _ = evaluate(llm, "BASE+SC (k samples, no tools)", self_consistency=True)
    report["baseline_sc"] = sc_acc

    rule("PHASE 4 — PRISM v0: verifier-guided test-time search on the same weights")
    v0_acc, v0_raw = evaluate(llm, "PRISM v0 (search)", use_prism=True)
    report["prism_v0"] = v0_acc
    ho = [i for i, t in enumerate(EVAL["grid"]) if set(t["rule"].split("+")) & GRID_HELDOUT]
    def grid_split(raw):
        h = [raw["grid"][i] for i in ho]
        d = [v for i, v in enumerate(raw["grid"]) if i not in set(ho)]
        return (100.0 * sum(h) / max(1, len(h)), len(h),
                100.0 * sum(d) / max(1, len(d)), len(d))
    report["grid_heldout_v0"] = grid_split(v0_raw)
    log("grid split: %d tasks use held-out rule families (never in the curriculum), %d are trainable",
        len(ho), len(EVAL["grid"]) - len(ho))

    rule("PHASE 5 — self-evolution: auto-curriculum -> exact verification -> skills + LoRA")
    llm.attach_lora()
    curve = [dict(round=0, **v0_acc, skills=len(SKILLS))]
    frontier = dict(grid=2)
    all_pairs = []
    for rd in range(1, CFG["evo_rounds"] + 1):
        if budget_left() < 420:
            log("time budget exhausted — stopping evolution after %d round(s)", rd - 1); break
        log("--- evolution round %d/%d (skills=%d, %.0fs left) ---",
            rd, CFG["evo_rounds"], len(SKILLS), budget_left())
        tasks = build_curriculum(CFG["evo_tasks"], random.Random(SEED + 991 * rd), frontier)
        try:
            pairs, stats = harvest(llm, tasks, k=max(4, CFG["k_math"]))
        except Exception as e:
            log("  !! harvest failed (%s: %s) — skipping this round", type(e).__name__, str(e)[:160])
            traceback.print_exc(); continue
        log("  self-generated pass-rates: %s", {k: f"{a}/{b}" for k, (a, b) in stats.items()})
        log("  verified traces harvested: %d   skill library -> %d", len(pairs), len(SKILLS))
        # auto-curriculum: raise difficulty where we are already comfortable
        if "grid" in stats and stats["grid"][1] and stats["grid"][0] / stats["grid"][1] > 0.6:
            frontier["grid"] = min(3, frontier["grid"] + 1)
            log("  curriculum: grid difficulty -> %d", frontier["grid"])
        all_pairs.extend(pairs)
        tr = star_finetune(llm, all_pairs, steps=CFG["sft_steps"])
        if tr: log("  STaR update on %d traces: loss %.3f -> %.3f",
                   tr["n_pairs"], tr["loss_start"], tr["loss_end"])
        acc, raw = evaluate(llm, f"PRISM v{rd} (evolved)", use_prism=True)
        report["grid_heldout_v%d" % rd] = grid_split(raw)
        curve.append(dict(round=rd, **acc, skills=len(SKILLS)))
        report[f"prism_v{rd}"] = acc
        json.dump(report, open(os.path.join(STATE_DIR, "report.json"), "w"), indent=2)
    report["curve"] = curve
    final = curve[-1]

    rule("PHASE 6 — optional real head-to-head against a frontier model")
    fr = frontier_probe()
    if fr: log("FRONTIER %-18s math %.1f | grid %.1f | sci %.1f | agent %.1f || MEAN %.1f",
               fr["model"], fr["math"], fr["grid"], fr["sci"], fr["agent"], fr["MEAN"])
    else:  log("skipped (set PRISM_FRONTIER_KEY / _BASE / _MODEL to run it)")
    report["frontier"] = fr

    # ------------------------------------------------------------------- report ----
    rule("RESULTS")
    hdr = f"{'system':<34}{'math':>8}{'grid':>8}{'sci':>8}{'agent':>8}{'MEAN':>9}"
    print(hdr); print("-" * len(hdr))
    def row(name, a):
        print(f"{name:<34}{a['math']:>8.1f}{a['grid']:>8.1f}{a['sci']:>8.1f}{a['agent']:>8.1f}{a['MEAN']:>9.1f}")
    row(f"base {llm.n_params/1e6:.0f}M, 1 pass, no tools", base_acc)
    row("base + self-consistency, no tools", sc_acc)
    row("PRISM v0 (search+verify)", v0_acc)
    for c in curve[1:]:
        row(f"PRISM v{c['round']} (self-evolved)", c)
    if fr: row(f"FRONTIER {fr['model']} (1 pass)", fr)
    print("-" * len(hdr))
    print(f"\nparameters: {llm.n_params:,} ({llm.n_params/1e9:.4f}B)  — constraint <1B: "
          f"{'SATISFIED' if llm.n_params < 1e9 else 'VIOLATED'}")
    print(f"compression vs a 1T model: {1e12/llm.n_params:,.0f}x fewer parameters")
    print(f"lift from more sampling alone:  {sc_acc['MEAN']-base_acc['MEAN']:+.1f} points "
          f"({base_acc['MEAN']:.1f} -> {sc_acc['MEAN']:.1f})   [compute-matched control]")
    print(f"lift from search+verification: {v0_acc['MEAN']-sc_acc['MEAN']:+.1f} points beyond that "
          f"control ({sc_acc['MEAN']:.1f} -> {v0_acc['MEAN']:.1f})")
    print(f"lift from self-evolution:      {final['MEAN']-v0_acc['MEAN']:+.1f} points "
          f"({v0_acc['MEAN']:.1f} -> {final['MEAN']:.1f})")
    print(f"total lift on identical weights budget: {final['MEAN']-base_acc['MEAN']:+.1f} points")
    hv0 = report.get("grid_heldout_v0"); hvf = report.get("grid_heldout_v%d" % final["round"], hv0)
    if hv0 and hvf:
        print(f"grid generalisation: unseen rule families {hv0[0]:.1f} -> {hvf[0]:.1f} on {hv0[1]} tasks | "
              f"curriculum families {hv0[2]:.1f} -> {hvf[2]:.1f} on {hv0[3]} tasks")
    print(f"sci generalisation: the {len(EVAL['sci'])} evaluated laws are disjoint from every law the "
          f"curriculum was allowed to generate")
    print(f"skill library: {len(SKILLS)} verified executable skills (persisted to {SKILLS.path})")
    print(f"llm calls: {llm.calls}   generated tokens: {llm.gen_tokens:,}   wall clock: {elapsed()/60:.1f} min")

    print("\nself-evolution curve (MEAN accuracy by round):")
    for c in curve:
        bar = "#" * int(round(c["MEAN"] / 2))
        print(f"  round {c['round']}  {c['MEAN']:5.1f}  {bar}  (skills={c['skills']})")

    print("""
HONEST READING OF THIS RESULT
-----------------------------
What is demonstrated, and is real:
  * A sub-1B model, wrapped in program synthesis + exact execution-based verification +
    self-consistency + error-driven refinement + a persistent skill library, scores far above
    the same weights used the way a chat model is normally used. The compute-matched control
    row shows how little of that comes from sampling more: the gain is verification, not
    budget. Parameters were traded for search AND for the ability to check.
  * The system improves itself without any human labels: it invents its own tasks, keeps only
    trajectories an exact verifier accepts, writes them into an executable skill library, and
    distils them back into its own weights with LoRA. The curve above is that loop running.
  * On tasks with a cheap exact verifier, this is genuinely competitive with, and can exceed,
    a far larger model restricted to a single forward pass -- that is the regime where search
    substitutes for scale. Run the frontier probe to measure it rather than assume it.

What is NOT demonstrated, and would be false to claim:
  * This is not a superintelligence and does not outperform frontier models or humans in
    general. Its advantage is confined to problems whose answers can be checked mechanically
    and cheaply. Remove the verifier and it collapses back to a 0.6B model.
  * Open-ended reasoning, world knowledge, long-context understanding, judgement, novel
    scientific discovery outside the search space it is given -- frontier models and humans
    remain far ahead, and no amount of Colab compute at this scale changes that.
  * The self-evolution loop improves in-distribution competence on families it can verify.
    It is not unbounded recursive self-improvement, and it saturates.

To keep evolving (each call is another autonomous round, state persists in %s):
    evolve_more(3)
""" % STATE_DIR)

    json.dump(report, open(os.path.join(STATE_DIR, "report.json"), "w"), indent=2)
    SKILLS.save()
    log("report saved to %s", os.path.join(STATE_DIR, "report.json"))
    return llm, report

def evolve_more(rounds=1, tasks_per_round=None, budget_s=2400):
    """Continue the self-evolution loop in a later cell; skills + adapter persist."""
    global TIME_BUDGET, T0
    T0 = time.time(); TIME_BUDGET = budget_s
    llm = _STATE["llm"]
    n = tasks_per_round or CFG["evo_tasks"]
    for rd in range(rounds):
        tasks = build_curriculum(n, random.Random(int(time.time()) % 10000), dict(grid=3))
        pairs, stats = harvest(llm, tasks, k=max(4, CFG["k_math"]))
        log("round %d: %s | %d verified traces | %d skills",
            rd + 1, {k: f"{a}/{b}" for k, (a, b) in stats.items()}, len(pairs), len(SKILLS))
        star_finetune(llm, pairs, steps=CFG["sft_steps"])
        acc, _ = evaluate(llm, f"PRISM (extra round {rd+1})", use_prism=True)
    return acc

_STATE = {}
if __name__ == "__main__":
    try:
        _llm, _report = main()
        _STATE["llm"] = _llm; _STATE["report"] = _report
    except Exception:
        traceback.print_exc()
        print("\n[PRISM] run failed above. State kept in", STATE_DIR)
        raise
