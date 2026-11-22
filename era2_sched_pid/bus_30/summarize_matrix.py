"""S9 matrix summary from the Simulink runs (matrix_<round>_<design>_<part>.mat). Writes matrix_summary.json."""
import glob, json, numpy as np, scipy.io as sio
S = sio.loadmat('matrix_scen.mat', squeeze_me=True); N = len(S['typ']); ne = int(S['n_eval'])
names = {1: 'gen', 2: 'load', 3: 'line'}; ty = np.array([names[int(t)] for t in S['typ']])
LIM = 0.005; out = {}
for rnd in 'AB':
    for des in ('pole', 'ceilA', 'ceilB'):
        fs = sorted(glob.glob(f'matrix_{rnd}_{des}_*.mat'))
        if not fs: continue
        pk = np.full(N, np.nan); ts = pk.copy(); tv = pk.copy()
        for f in fs:
            m = sio.loadmat(f, squeeze_me=True); i = np.atleast_1d(m['idx']) - 1
            pk[i], ts[i], tv[i] = m['peak'], m['ts'], m['tv']
        if np.isnan(pk).any(): continue
        e = slice(0, ne); ok = pk[e] <= LIM; tye = ty[e]
        row = dict(n=ne, in_envelope=float(ok.mean()),
                   by_type={t: float(ok[tye == t].mean()) for t in ['gen', 'load', 'line']},
                   gen_G1=float(ok[(tye == 'gen') & (S['loc'][e] == 1)].mean()), gen_G2=float(ok[(tye == 'gen') & (S['loc'][e] == 2)].mean()),
                   max_dev_median={t: float(np.median(pk[e][tye == t])) for t in ['gen', 'load', 'line']},
                   max_dev_p95={t: float(np.percentile(pk[e][tye == t], 95)) for t in ['gen', 'load', 'line']},
                   ts_median={t: float(np.median(ts[e][tye == t])) for t in ['gen', 'load', 'line']},
                   tv_median={t: float(np.median(tv[e][tye == t])) for t in ['gen', 'load', 'line']})
        r = slice(ne, N); a = S['alpha'][r]; d = S['tc'][r] - S['tf'][r]; okr = pk[r] <= LIM
        row['G1_alpha_reach'] = {f'{dd:.1f}s': (float(a[(np.isclose(d, dd)) & okr].max()) if ((np.isclose(d, dd)) & okr).any() else None) for dd in (0.1, 0.5, 3.0)}
        out[f'{rnd}/{des}'] = row
        print(f"{rnd}/{des:5s}: all {row['in_envelope']:.3f} | gen {row['by_type']['gen']:.3f} (G1 {row['gen_G1']:.3f}, G2 {row['gen_G2']:.3f}) load {row['by_type']['load']:.3f} line {row['by_type']['line']:.3f}"
              f" | maxdev med g/l/l {row['max_dev_median']['gen']:.4f}/{row['max_dev_median']['load']:.4f}/{row['max_dev_median']['line']:.5f}"
              f" | ts med {row['ts_median']['gen']:.2f}/{row['ts_median']['load']:.2f}/{row['ts_median']['line']:.2f} | TV med {row['tv_median']['gen']:.2f}/{row['tv_median']['load']:.2f}/{row['tv_median']['line']:.2f} | G1 reach {row['G1_alpha_reach']}")
json.dump(out, open('matrix_summary.json', 'w'), indent=1)
