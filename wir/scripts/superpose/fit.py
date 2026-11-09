# Fit lossy superposition bottlenecks for the ant brain on recorded colony states and
# emit WIR programs. python fit.py rec.json ant_brain.wir outdir m1,m2,...
import json, sys, re, numpy as np, torch
rec, src_p, out, ms = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')]
R = json.load(open(rec)); names = R['inputs']
X = np.array([s for s, _ in R['states']], float); cnt = np.array([n for _, n in R['states']], float)
v = {n: X[:, i] for i, n in enumerate(names)}
b = lambda x: (x > 0.5).astype(float)
cf, cs, gr = v['cf'], v['cs'], v['gr']
free = (1 - cf) * (1 - cs) * (1 - gr)
hungry = np.clip(v['hunger'] - 5, 0, 10); starving = b(v['hunger'] >= 12); scared = b(v['fs'] >= 1); hs = b(v['store'] >= 1)
kn, kf, kg, kh = v['kn'], v['kf'], v['kg'], v['kh']
idle = free * (1 - kf) * (1 - kg) * (1 - kh)
home = kn * (6 * cf + hs * 2 * np.minimum(hungry, 4) + 4 * free * b(v['dns'] >= 60))
fb = free * kf
forage = fb * (5 - v['store'] + 2 * v['old'] + 3 * v['r1']) + fb * (1 - hs) * np.minimum(hungry, 4)
dig = free * kg * (1 - starving) * (4 - 2 * v['old'] + 3 * v['r0'])
dump = 6 * cs * kn
help_ = free * kh * (1 - starving) * (3 + 3 * v['r2'])
lost = np.maximum(idle, 1 - kn)
dig2 = free * v['ld'] * (1 - starving) * (1 - scared)
grip = np.maximum(gr, free * v['lp']) * (1 - starving) * (1 - scared)
workem = np.maximum.reduce([cs, dig2, free * kg])
FN = ['home', 'forage', 'dig', 'dump', 'help', 'lost', 'hungry', 'workem', 'cf', 'grip']
F = np.stack([home, forage, dig, dump, help_, lost, hungry, workem, cf, grip], 1)
# dedupe on (features, scared)
key = np.concatenate([F, scared[:, None]], 1)
uk, inv = np.unique(key, axis=0, return_inverse=True)
w = np.bincount(inv.ravel(), weights=cnt); F, sc = uk[:, :10], uk[:, 10]
print(f'{len(X)} recorded states, {cnt.sum():.0f} decisions, {len(F)} distinct feature vectors')
Ft, sct, wt = map(lambda a: torch.tensor(a, dtype=torch.float64), (F, sc, w / w.sum()))
def keys(G, s):
    t = torch.stack([7*G[:,0]+6, 7*G[:,1]+5, 7*G[:,2]+4, 7*G[:,3]+3, 7*G[:,4]+2, 21*G[:,5]+1, 56*s], 1)
    e = torch.stack([torch.full_like(s, 13.), 7*G[:,6]+5, 21*G[:,7]+4, 28*G[:,8]+3, 35*G[:,9]+2, 63*s+1, 14*G[:,5]], 1)
    return t, e
T0, E0 = keys(Ft, sct); tt, et = T0.argmax(1), E0.argmax(1)
scale = torch.tensor(F.max(0).clip(1), dtype=torch.float64)
def acc(C, c, D, d):
    G = torch.relu(Ft @ C.T + c) @ D.T + d
    T, E = keys(G, sct)
    return float((wt * (T.argmax(1) == tt)).sum()), float((wt * (E.argmax(1) == et)).sum()), G
summary = []
for m in ms:
    best = None
    for restart in range(6):
        torch.manual_seed(restart)
        C = (torch.randn(m, 10, dtype=torch.float64) * 0.5).requires_grad_()
        c = torch.zeros(m, dtype=torch.float64).requires_grad_()
        D = (torch.randn(10, m, dtype=torch.float64) * 0.5).requires_grad_()
        d = torch.zeros(10, dtype=torch.float64).requires_grad_()
        opt = torch.optim.Adam([C, c, D, d], lr=0.02)
        for it in range(3000):
            G = torch.relu((Ft / scale) @ C.T + c) @ D.T * scale + d
            T, E = keys(G, sct)
            loss = (wt * (torch.nn.functional.cross_entropy(T, tt, reduction='none') + torch.nn.functional.cross_entropy(E, et, reduction='none'))).sum() \
                 + 0.01 * (wt[:, None] * ((G - Ft) / scale) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Cr = torch.round(C / scale * 100) / 100; cr = torch.round(c * 100) / 100
            Dr = torch.round(D * scale[:, None] * 100) / 100; dr = torch.round(d * 100) / 100
            ta, ea, G = acc(Cr, cr, Dr, dr)
        if best is None or ta + ea > best[0]: best = (ta + ea, ta, ea, Cr, cr, Dr, dr, G)
    _, ta, ea, C, c, D, d, G = best
    # polysemanticity: features each neuron correlates with (|r| > 0.3, decision-weighted)
    H = torch.relu(Ft @ C.T + c).numpy(); ww = w / w.sum()
    def wcorr(a, b_):
        am, bm = (ww * a).sum(), (ww * b_).sum(); cov = (ww * (a - am) * (b_ - bm)).sum()
        sa, sb = np.sqrt((ww * (a - am) ** 2).sum()), np.sqrt((ww * (b_ - bm) ** 2).sum())
        return 0 if sa == 0 or sb == 0 else cov / sa / sb
    poly = [[FN[i] for i in range(10) if abs(wcorr(H[:, j], F[:, i])) > 0.3] for j in range(m)]
    print(f'm={m}: task acc {100*ta:.3f}%  emote acc {100*ea:.3f}%  (decision-weighted, rounded weights)')
    for j in range(m): print(f'   z{j}: {", ".join(poly[j]) or "-"}')
    summary.append({'m': m, 'task': ta, 'emote': ea, 'poly': poly})
    # emit WIR
    src = open(src_p).read()
    fx = {'home': 'home', 'forage': 'forage', 'dig': 'dig', 'dump': 'dump', 'help': 'help', 'lost': 'or(idle, not(kn))',
          'hungry': 'hungry', 'workem': 'or(cs, dig2, and(free, kg))', 'cf': 'cf', 'grip': 'grip'}
    fmt = lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')
    lin = lambda coefs, terms, k0: ' + '.join([fmt(k0)] + [f'{fmt(k)}*{t}' for k, t in zip(coefs, terms) if abs(k) > 1e-9]).replace('+ -', '- ')
    L = ['# --- superposition bottleneck: ten features share %d neurons ---' % m]
    L.append('let sp_lost = or(idle, not(kn))'); L.append('let sp_work = or(cs, dig2, and(free, kg))')
    terms = [fx[n] if n not in ('lost', 'workem') else ('sp_lost' if n == 'lost' else 'sp_work') for n in FN]
    for j in range(m): L.append(f'let z{j} = unit(relu({lin(C[j].tolist(), terms, float(c[j]))}))')
    for i, n in enumerate(FN): L.append(f'let r_{n} = {lin(D[i].tolist(), [f"z{j}" for j in range(m)], float(d[i]))}')
    block = '\n'.join(L) + '\n'
    sub = {r'let tkey0 = 7\*home \+ 6': 'let tkey0 = 7*r_home + 6', r'let tkey1 = 7\*forage \+ 5': 'let tkey1 = 7*r_forage + 5',
           r'let tkey2 = 7\*dig \+ 4': 'let tkey2 = 7*r_dig + 4', r'let tkey3 = 7\*dump \+ 3': 'let tkey3 = 7*r_dump + 3',
           r'let tkey4 = 7\*help \+ 2': 'let tkey4 = 7*r_help + 2', r'let tkey5 = 7\*explore \+ 1': 'let tkey5 = 21*r_lost + 1',
           r'let ekey1 = 7\*\(hungry\) \+ 5': 'let ekey1 = 7*r_hungry + 5',
           r'let ekey2 = 7\*\(3\*or\(cs, dig2, and\(free, kg\)\)\) \+ 4': 'let ekey2 = 21*r_workem + 4',
           r'let ekey3 = 7\*\(4\*cf\) \+ 3': 'let ekey3 = 28*r_cf + 3', r'let ekey4 = 7\*\(5\*grip\) \+ 2': 'let ekey4 = 35*r_grip + 2',
           r'let ekey6 = 7\*\(2\*or\(idle, not\(kn\)\)\) \+ 0': 'let ekey6 = 14*r_lost'}
    for pat, rep in sub.items():
        src, n = re.subn(pat, rep, src); assert n == 1, pat
    src = src.replace('let tkey0 =', block + 'let tkey0 =', 1)
    open(f'{out}/sp{m}.wir', 'w').write(src)
json.dump(summary, open(f'{out}/fit_summary.json', 'w'), indent=1)
