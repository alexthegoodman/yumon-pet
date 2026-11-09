import json, sys, numpy as np
S=sys.argv[1]
def load(n):
    d=json.load(open(S+'/'+n+'.json'))
    L=[(np.array(l['W'] if 'W' in l else l['weight'],float), np.array(l['b'] if 'b' in l else l['bias'],float)) for l in d['layers']]
    return d['inputs'], d['outputs'], L
io,oo,Lo=load('orig'); ip,op,Lp=load('poly')
assert io==ip and oo==op, (io,ip,oo,op)
nz=lambda L: sum(int((W!=0).sum()+(b!=0).sum()) for W,b in L)
print('nonzero orig',nz(Lo),'poly',nz(Lp))
rng=np.random.default_rng(1); N=200000
cols={}
def oh(k): x=np.zeros((N,k)); x[np.arange(N),rng.integers(0,k,N)]=1; return x
o=oh(5); w=oh(4)
for i in range(5): cols[f'o{i}']=o[:,i]
for i in range(4): cols[f'w{i}']=w[:,i]
carry=rng.integers(0,4,N)  # 0 none,1 cf,2 cs,3 gr
cols['cf']=(carry==1)*1.;cols['cs']=(carry==2)*1.;cols['gr']=(carry==3)*1.
cols['hunger']=rng.integers(0,16,N)*1.;cols['store']=rng.integers(0,4,N)*1.;cols['fs']=rng.integers(0,4,N)*1.;cols['dns']=rng.integers(0,121,N)*1.
for nme in io:
    if nme not in cols:
        if nme[:2] in('gn','gf','gg','gm','gh','fg'): cols[nme]=rng.integers(-1,2,N)*1.
        else: cols[nme]=rng.integers(0,2,N)*1.
X=np.stack([cols[n] for n in io],1)
def run(L,X):
    h=X
    for i,(W,b) in enumerate(L):
        h=h@W.T+b
        if i<len(L)-1: h=np.maximum(h,0)
    return h
A=run(Lo,X);B=run(Lp,X)
bad=np.abs(A-B)>1e-6
print('rows differing',bad.any(1).mean())
for j,n in enumerate(oo):
    if bad[:,j].any(): print(' ',n,'differs on',bad[:,j].mean(), 'example rows', np.where(bad[:,j])[0][:3])
