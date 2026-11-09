# No-training codes for the ant bottleneck: rank of the real states, PCA (closed form),
# and fixed encoders (PCA / random / designed) with the decoder SOLVED by a linear program
# whose constraints are the decisions themselves (margin >= 1 => compiled gt bits exact).
import json, sys, numpy as np
from scipy.optimize import linprog
S = sys.argv[1]
R = json.load(open(S + '/rec7.json')); names = R['inputs']
X = np.array([s for s, _ in R['states']], float); cnt = np.array([n for _, n in R['states']], float)
v = {n: X[:, i] for i, n in enumerate(names)}
b = lambda x: (x > 0.5).astype(float)
cf, cs, gr = v['cf'], v['cs'], v['gr']; free = (1-cf)*(1-cs)*(1-gr)
hungry = np.clip(v['hunger']-5, 0, 10); starving = b(v['hunger'] >= 12); scared = b(v['fs'] >= 1); hs = b(v['store'] >= 1)
kn, kf, kg, kh = v['kn'], v['kf'], v['kg'], v['kh']; idle = free*(1-kf)*(1-kg)*(1-kh)
home = kn*(6*cf + hs*2*np.minimum(hungry, 4) + 4*free*b(v['dns'] >= 60)); fb = free*kf
forage = fb*(5 - v['store'] + 2*v['old'] + 3*v['r1']) + fb*(1-hs)*np.minimum(hungry, 4)
dig = free*kg*(1-starving)*(4 - 2*v['old'] + 3*v['r0']); dump = 6*cs*kn
help_ = free*kh*(1-starving)*(3 + 3*v['r2']); lost = np.maximum(idle, 1-kn)
dig2 = free*v['ld']*(1-starving)*(1-scared); grip = np.maximum(gr, free*v['lp'])*(1-starving)*(1-scared)
workem = np.maximum.reduce([cs, dig2, free*kg])
FN = ['home','forage','dig','dump','help','lost','hungry','workem','cf','grip']
F = np.stack([home, forage, dig, dump, help_, lost, hungry, workem, cf, grip], 1)
key = np.concatenate([F, scared[:, None]], 1)
uk, inv = np.unique(key, axis=0, return_inverse=True)
w = np.bincount(inv.ravel(), weights=cnt); w = w / w.sum(); F, sc = uk[:, :10], uk[:, 10]; N = len(F)
# keys as affine maps of r (10 features): K = r @ A.T + k0 (A: 7x10), exact parts in k0
At = np.zeros((7, 10)); kt = np.zeros((N, 7))
for i in range(5): At[i, i] = 7; kt[:, i] = 6 - i
At[5, 5] = 21; kt[:, 5] = 1; kt[:, 6] = 56*sc
Ae = np.zeros((7, 10)); ke = np.zeros((N, 7)); ke[:, 0] = 13
Ae[1, 6] = 7; ke[:, 1] = 5; Ae[2, 7] = 21; ke[:, 2] = 4; Ae[3, 8] = 28; ke[:, 3] = 3; Ae[4, 9] = 35; ke[:, 4] = 2
ke[:, 5] = 63*sc + 1; Ae[6, 5] = 14
tt = (F @ At.T + kt).argmax(1); et = (F @ Ae.T + ke).argmax(1)
print(f'{N} distinct states; {len(set(zip(tt, et)))} distinct (task, emote) decisions')
# 1. how many dimensions do the real states use?
Fs = F @ np.diag([7,7,7,7,7,21,7,21,28,35])  # key scale
mu = (w[:, None]*Fs).sum(0); sv = np.linalg.svd(np.sqrt(w)[:, None]*(Fs-mu), compute_uv=False)
print('affine rank of the states:', int((sv > 1e-9*sv[0]).sum()), ' energy kept by top m:', ' '.join(f'{m}:{100*(sv[:m]**2).sum()/(sv**2).sum():.1f}%' for m in range(1, 11)))
def score(Z, D, d):
    r = Z @ D.T + d; T = r @ At.T + kt; E = r @ Ae.T + ke
    ok_t = T.argmax(1) == tt; ok_e = E.argmax(1) == et
    mt = np.sort(T, 1); me = np.sort(E, 1)
    clean = ok_t & ok_e & (mt[:, -1]-mt[:, -2] >= 1-1e-9) & (me[:, -1]-me[:, -2] >= 1-1e-9)
    return 100*(w*ok_t).sum(), 100*(w*ok_e).sum(), int((~clean).sum())
def lp_decoder(Z, margin=1.0):
    m = Z.shape[1]; nv = 10*(m+1)  # D (10 x m) row-major, then d (10)
    rows, rhs, slack_of = [], [], []
    for s in range(N):
        zs = np.append(Z[s], 1.0)
        for A, k0, y, tag in ((At, kt, tt, 0), (Ae, ke, et, 1)):
            for j in range(7):
                if j == y[s]: continue
                # (A[y]-A[j]) . r + k0[y]-k0[j] >= margin - slack ; r_i = sum_c D[i,c] z_c + d_i
                g = A[y[s]] - A[j]; row = np.zeros(nv)
                for i in np.nonzero(g)[0]:
                    row[i*m:(i+1)*m] += g[i]*Z[s]; row[10*m+i] += g[i]
                rows.append(-row); rhs.append(-(margin - (k0[s, y[s]] - k0[s, j]))); slack_of.append(2*s+tag)
    rows = np.array(rows); ns = 2*N
    Aub = np.zeros((len(rows), nv+ns)); Aub[:, :nv] = rows; Aub[np.arange(len(rows)), nv+np.array(slack_of)] = -1
    c = np.concatenate([np.zeros(nv), np.repeat(w, 2) + 1e-6])
    res = linprog(c, A_ub=Aub, b_ub=np.array(rhs), bounds=[(-60, 60)]*nv + [(0, None)]*ns, method='highs')
    x = res.x; return x[:10*m].reshape(10, m), x[10*m:nv]
U = np.linalg.svd(np.sqrt(w)[:, None]*(Fs-mu), full_matrices=False)[2]
rng = np.random.default_rng(0)
for m in (6, 5, 4, 3):
    # PCA encoder in key scale; unit() keeps it linear (shifted so relu never clips)
    Zp = (Fs - mu) @ U[:m].T; Zp = Zp - Zp.min(0)
    Dp, dp = np.linalg.lstsq(np.c_[Zp, np.ones(N)] * np.sqrt(w)[:, None], F * np.sqrt(w)[:, None], rcond=None)[0][:m].T, None
    sol = np.linalg.lstsq(np.c_[Zp, np.ones(N)] * np.sqrt(w)[:, None], F * np.sqrt(w)[:, None], rcond=None)[0]
    a = score(Zp, sol[:m].T, sol[m])
    D, d = lp_decoder(Zp); b_ = score(Zp, D, d)
    # random relu encoder (no design at all) + LP decoder
    Cr = rng.normal(size=(m, 10)) / F.max(0).clip(1); cr = rng.normal(size=m)*0.5
    Zr = np.maximum(F @ Cr.T + cr, 0); D2, d2 = lp_decoder(Zr); c_ = score(Zr, D2, d2)
    print(f'm={m}: PCA+least squares task {a[0]:.3f}% emote {a[1]:.3f}% bad states {a[2]}/{N} | '
          f'PCA+LP {b_[0]:.3f}% {b_[1]:.3f}% bad {b_[2]} | random relu+LP {c_[0]:.3f}% {c_[1]:.3f}% bad {c_[2]}')
