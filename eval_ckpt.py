"""Held-out plain CE + sampled-text repetition for one checkpoint (CPU). MODEL_SET and EVAL_LOOPS env vars select the size and loop count.
    MODEL_SET="dim=768,..." EVAL_LOOPS=8 python eval_ckpt.py <checkpoint.pt> <outdir>"""
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F, tiktoken
from anthos.configs import get_training_config
from anthos.main import Anthos
ck, outdir = sys.argv[1], sys.argv[2]
DEV = "cpu"   # training holds ~all GPU memory; evaluating on the GPU runs it out of memory
torch.set_num_threads(16)
cfg, _ = get_training_config("proof"); cfg.vocab_size = 100352; cfg.thought_prefix_only = True; cfg.moe_fix_weight_order = True
for kv in os.environ.get("MODEL_SET", "").split(","):    # e.g. dim=768,n_heads=12,... so one script serves every size
    if kv:
        k, v = kv.split("="); setattr(cfg, k, type(getattr(cfg, k))(v))
m = Anthos(cfg).to(DEV).eval(); sd = torch.load(ck, map_location="cpu", weights_only=False); m.load_state_dict(sd["model"], strict=False)
val = np.memmap("data/pretrain_full/val.bin", dtype=np.uint32, mode="r")
tr = np.memmap("data/pretrain_full/train_00000.bin", dtype=np.uint32, mode="r")[:50_000_000]
p = (np.bincount(tr, minlength=100352) + 1.0); p /= p.sum()
rng = np.random.default_rng(0); L, P = 512, 64; ce, uni = [], []
for _ in range(32):
    s = int(rng.integers(0, len(val) - L - 1)); w = torch.tensor(val[s:s + L + 1].astype(np.int64))[None]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        lg, _ = m(w[:, :-1], n_loops=int(os.environ.get("EVAL_LOOPS", "8")), return_aux=True, prefix_len=torch.tensor([P]))
    t = w[:, 1:]; ce.append(F.cross_entropy(lg[0, P - 1:].float(), t[0, P - 1:]).item()); uni.append(float(-np.log(p[t[0, P - 1:].cpu().numpy()]).mean()))
enc = tiktoken.get_encoding("cl100k_base"); torch.manual_seed(0)
def rep(ids, n=4):
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]; return 1 - len(set(g)) / max(1, len(g))
prompts = ["The water cycle is the process by which", "In 1066, William of Normandy", "To solve a quadratic equation, you can", "Photosynthesis is important because"]
txt, reps = [], []
for pr in prompts:
    ids = enc.encode_ordinary(pr); x = torch.tensor([ids]); cache = {}; gen = []
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x, n_loops=int(os.environ.get("EVAL_LOOPS", "8")), kv_cache=cache, prefix_len=len(ids))[0, -1]
        for j in range(100):
            q = torch.softmax(out.float() / 0.8, -1); top = q.topk(40); nxt = int(top.indices[torch.multinomial(top.values, 1)])
            gen.append(nxt); out = m(torch.tensor([[nxt]]), n_loops=int(os.environ.get("EVAL_LOOPS", "8")), kv_cache=cache, start_pos=len(ids) + j, prefix_len=len(ids))[0, -1]
    reps.append(rep(gen)); txt.append(f"### {pr}\n{pr}{enc.decode(gen)}\n")
step = sd.get("step")
open(f"{outdir}/samples_{step:06d}.txt", "w").write("\n".join(txt))
print(json.dumps({"step": step, "ce": round(float(np.mean(ce)), 4), "unigram": round(float(np.mean(uni)), 3), "sample_rep": round(float(np.mean(reps)), 3)}))
