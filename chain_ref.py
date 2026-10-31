"""Reference check: can an ordinary PyTorch transformer learn the 1-hop lookup in chain_probe.py in this many steps?
If it can and the Anthos regular model cannot, the Anthos code path is the problem; if neither can, the task is just slow to learn.

    python chain_ref.py --layers 3 --steps 1500 --lr 1e-3
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from chain_probe import N_NODES, NODE0, make_batch

VOCAB = 8192


class Ref(nn.Module):
    def __init__(self, d=192, heads=6, layers=3, max_len=64):
        super().__init__()
        self.emb, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc, self.head = nn.TransformerEncoder(layer, layers), nn.Linear(d, VOCAB)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return self.head(self.enc(h, mask=mask, is_causal=True))[:, -1, NODE0:NODE0 + N_NODES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    m = Ref(layers=a.layers)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 50)
        seq, ans, _ = make_batch(a.batch, gen, max_hops=1)
        loss = F.cross_entropy(m(seq), ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                s, t, _ = make_batch(512, torch.Generator().manual_seed(123), hops=1)
                acc = (m(s).argmax(-1) == t).float().mean().item()
            print(json.dumps({"step": step, "loss": round(loss.item(), 3), "k1": round(acc, 3)}), flush=True)


if __name__ == "__main__":
    main()
