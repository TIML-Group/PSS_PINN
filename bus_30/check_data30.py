"""
check_data30.py
===============
Cheap sanity check on the converted memmap, to run BEFORE training.

    python check_data30.py --data data30_mm

The one silent failure mode in a sharded conversion is misalignment: row k of
X holding a different scenario than label k. Training would run happily and
learn nothing. The labels give a test for it -- `y_ool` says whether
|omega-1| ever left the envelope, and that is measurable straight from X. If
the rows were shuffled, agreement would collapse to chance.

Also checks: no all-zero rows (an uncovered scenario), a flat pre-fault start,
and that omega noise matches noise_std.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data30_mm')
ap.add_argument('--n', type=int, default=2000, help='scenarios to check')
args = ap.parse_args()

X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(args.data, 'meta.npz'), allow_pickle=True)
g = lambda k: md[k].ravel()

N, T, C = X.shape
NG = int(md['n_machines']) if 'n_machines' in md else 6
W_IDX = list(range(1, 2 * NG, 2))
t = g('t')
limit = float(md['omega_limit'])
y_ool = g('y_ool').astype(bool)
y_type = g('y_type').astype(int)
t_fault = g('y_t_fault')
print(f"X {X.shape}  {X.nbytes/2**30:.2f} GB   limit {limit}")

ks = np.linspace(0, N - 1, min(args.n, N)).astype(int)
ool_meas = np.zeros(len(ks), bool)
zero_rows, flat_start, noise = [], [], []
for i, k in enumerate(ks):
    w = np.asarray(X[k][:, W_IDX])
    if not np.any(X[k]):
        zero_rows.append(int(k)); continue
    ool_meas[i] = bool((np.abs(w - 1) > limit).any())
    i_f = max(1, int(np.searchsorted(t, t_fault[k])) - 2)
    pre = w[:i_f]
    if len(pre) > 10:
        flat_start.append(float(np.abs(pre.mean(0) - 1).max()))
        noise.append(float(pre.std(0).mean()))

agree = (ool_meas == y_ool[ks]).mean()
print(f"\nOOL from X vs y_ool label: {agree:.4f} agreement on {len(ks)} scenarios")
print(f"  (chance would be ~{max(y_ool.mean(), 1-y_ool.mean()):.2f}; "
      f"a shuffled conversion lands there)")
dis = np.where(ool_meas != y_ool[ks])[0]
if len(dis):
    print(f"  {len(dis)} disagreements -- peaks near the limit are expected, "
          f"since the memmap is decimated to 10 ms and the label was computed "
          f"at 1 ms:")
    for i in dis[:6]:
        k = ks[i]
        w = np.asarray(X[k][:, W_IDX])
        print(f"    scenario {k:5d} type {y_type[k]} label {y_ool[k]} "
              f"measured peak {np.abs(w-1).max():.5f}")

print(f"\npre-fault |mean(omega)-1|: max {max(flat_start):.2e}  "
      f"(should sit at the noise level, not drift)")
print(f"pre-fault omega std: {np.mean(noise):.2e}  "
      f"(noise_std says {g('noise_std')[1]:.2e})")
if zero_rows:
    print(f"\nWARNING: {len(zero_rows)} all-zero scenarios (not covered by any "
          f"shard): {zero_rows[:10]}")
else:
    print("\nno all-zero scenarios")
cov = md['scenario_covered'] if 'scenario_covered' in md else np.ones(N, bool)
print(f"covered by a shard: {int(np.sum(cov))}/{N}")
