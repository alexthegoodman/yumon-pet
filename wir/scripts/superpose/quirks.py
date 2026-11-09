import json, sys, numpy as np
from collections import Counter
S = sys.argv[1]
R = json.load(open(S + '/rec7.json')); names = R['inputs']
X = np.array([s for s, _ in R['states']], float); cnt = np.array([n for _, n in R['states']], float)
def load(p):
    d = json.load(open(p)); return d['outputs'], [(np.array(l['W']), np.array(l['b'])) for l in d['layers']]
def run(L, X):
    h = X
    for i, (W, b) in enumerate(L):
        h = h @ W.T + b
        if i < len(L) - 1: h = np.maximum(h, 0)
    return h
outs, Lo = load(S + '/orig.json')
O = run(Lo, X)
EM = ['calm', 'hungry', 'working', 'carrying', 'straining', 'scared', 'curious']; OP = ['E', 'N', 'W', 'S', 'hold']
oi = lambda n: outs.index(n)
def dec(Y):
    em = np.zeros(len(Y), int); op = np.zeros(len(Y), int)
    for k in range(7): em[Y[:, oi(f'em{k}')] > 0.5] = k
    for k in range(5): op[Y[:, oi(f'o2_{k}')] > 0.5] = k
    return em, op
em0, op0 = dec(O)
v = {n: X[:, i] for i, n in enumerate(names)}
ctx = np.where(v['cf'] > 0, 'carrying food', np.where(v['cs'] > 0, 'carrying sand', np.where(v['gr'] > 0, 'gripping', 'free')))
hun = np.where(v['hunger'] >= 12, 'starving', np.where(v['hunger'] >= 6, 'hungry', 'fed'))
tot = cnt.sum()
for m in sys.argv[2].split(','):
    _, L = load(f'{S}/sp{m}.json'); Y = run(L, X); em, op = dec(Y)
    print(f'\n=== m={m}: emote differs {100*cnt[em!=em0].sum()/tot:.2f}%, heading differs {100*cnt[op!=op0].sum()/tot:.2f}% of decisions')
    c = Counter()
    for i in np.where(em != em0)[0]: c[f'{EM[em0[i]]} -> {EM[em[i]]}'] += cnt[i]
    print('  emote swaps:', '; '.join(f'{k} {100*n/tot:.2f}%' for k, n in c.most_common(5)))
    c = Counter()
    for i in np.where(op != op0)[0]: c[f'{hun[i]}, {ctx[i]}, {"nest" if v["kn"][i] else "away"}'] += cnt[i]
    print('  heading changes happen when:', '; '.join(f'{k} {100*n/tot:.2f}%' for k, n in c.most_common(5)))
    hold = cnt[(op != op0) & (op == 4)].sum(); unhold = cnt[(op != op0) & (op0 == 4)].sum()
    print(f'  of heading changes: now holds {100*hold/max(1,cnt[op!=op0].sum()):.0f}%, stops holding {100*unhold/max(1,cnt[op!=op0].sum()):.0f}%')
