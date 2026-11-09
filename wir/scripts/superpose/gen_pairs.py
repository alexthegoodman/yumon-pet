# Variant 2: two ternary grades per neuron (base 3, values 0..8). The task picks one packed
# neuron per pair (contextual meaning), decoded in one layer.
import re, sys
src = open(sys.argv[1]).read()
fields = [('gn0','gn1','gn2','gn3'),('gf0','gf1','gf2','gf3'),('gg0','gg1','gg2','gg3'),
          ('gm0','gm1','gm2','gm3'),('gh0','gh1','gh2','gh3'),('w0','w1','w2','w3'),('fg0','fg1','fg2','fg3')]
L = []
for i,f in enumerate(fields):
    L.append(f"let pa{i} = unit(3*{f[0]} + {f[1]} + 4)")
    L.append(f"let pb{i} = unit(3*{f[2]} + {f[3]} + 4)")
for h in 'ab':
    L.append(f"let pk{h} = " + " + ".join(f"select(task{i}, p{h}{i}, 0)" for i in range(7)))
    L.append(f"let hi{h} = ge(pk{h}, 3) + ge(pk{h}, 6)")
    L.append(f"let lo{h} = pk{h} - 3*hi{h}")
for k,d in enumerate(['hia','loa','hib','lob']):
    L.append(f"let s{k} = 4*({d} - 1) + k{k}")
out, n = re.subn(r"let s0 = .*?\nlet s3 = [^\n]*\n", "\n".join(L) + "\n", src, flags=re.S)
assert n == 1
open(sys.argv[2],'w').write(out)
