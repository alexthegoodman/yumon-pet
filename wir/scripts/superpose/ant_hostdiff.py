# Offline: decode each compiled ant brain like the host (last bit > 0.5) on recorded states,
# report decision-weighted heading / emote / action differences vs the exact brain.
import json, sys, numpy as np
R = json.load(open(sys.argv[1])); X = np.array([s for s, _ in R['states']], float); cnt = np.array([n for _, n in R['states']], float)
def load(p):
    d = json.load(open(p)); return d['outputs'], [(np.array(l['W']), np.array(l['b'])) for l in d['layers']]
def run(L):
    h = X
    for i, (W, b) in enumerate(L):
        h = h @ W.T + b
        if i < len(L) - 1: h = np.maximum(h, 0)
    return h
def dec(outs, Y):
    o = np.zeros(len(Y), int); e = np.zeros(len(Y), int)
    for k in range(5): o[Y[:, outs.index(f'o2_{k}')] > 0.5] = k
    for k in range(7): e[Y[:, outs.index(f'em{k}')] > 0.5] = k
    a = np.stack([Y[:, outs.index(n)] > 0.5 for n in ['pick','dig2','grip','drop','eat']], 1)
    return o, e, a
outs, L0 = load(sys.argv[2]); o0, e0, a0 = dec(outs, run(L0)); tot = cnt.sum()
for p in sys.argv[3:]:
    outs, L = load(p); o, e, a = dec(outs, run(L))
    print(f'{p.split("/")[-1]:14s} heading {100*cnt[o!=o0].sum()/tot:.3f}%  emote {100*cnt[e!=e0].sum()/tot:.3f}%  actions {100*cnt[(a!=a0).any(1)].sum()/tot:.3f}%')
