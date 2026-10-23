import json
o = json.load(open('receding_cascade_ab/refcg_ab_off/refine_pareto.json'))
n = json.load(open('receding_cascade_ab/refcg_ab_on/refine_pareto.json'))
maxdr = maxdn = lenmax = 0.0
same = len(o) == len(n)
for a, b in zip(o, n):
    maxdr = max(maxdr, abs(a['rescue'] - b['rescue']))
    maxdn = max(maxdn, abs(a['norm'] - b['norm']))
    lam_d = abs(a['lam'] - b['lam']); lenmax = max(lenmax, lam_d)
    print(f"ep {a['epoch']}: eager {a['rescue']:.6f}/{a['norm']:.6f}  CG {b['rescue']:.6f}/{b['norm']:.6f}  d={a['rescue']-b['rescue']:+.2e}/{a['norm']-b['norm']:+.2e}  lam d={a['lam']-b['lam']:+.2e}")
print('same epoch count:', same)
print('MAX DIFF  rescue:', maxdr, ' norm:', maxdn, ' lam:', lenmax)
print('VERDICT:', 'IDENTICAL-within-float-tol' if (maxdr < 1e-9 and maxdn < 1e-9) else 'MISMATCH — investigate')
