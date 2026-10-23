"""
convert_dataset.py
==================
One-time conversion of pinn_dataset.mat into a scenario-major memmap plus a
small metadata file.

    python convert_dataset.py --mat pinn_dataset.mat --out data_mm

Why this is needed
------------------
The .mat stores X as (33, 8001, N) with the scenario index on the FASTEST
varying axis, so reading one scenario's window touches 33*8001 locations each
N*4 bytes apart -- the worst possible access pattern, and it gets worse as the
dataset grows. The converter rewrites it as (N, 8001, 33) in C order, so one
scenario's time-block is contiguous and a window read is a single strided slice
of a few hundred KB.

float32 is kept deliberately. float16 would halve the file, but omega sits at
~1.0 with deviations of 0.002-0.02 and the out-of-limit threshold is 0.005;
float16 spacing near 1.0 is about 0.001, which would corrupt the single most
important signal in the dataset. 10.6 GB on an NVMe with 31 GB of RAM for page
cache is the right trade.

Segmented files: pass several --mat arguments and they are concatenated in the
order given. Scenario indices in the output run over the concatenation.
"""

import argparse, json, os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--mat', type=str, nargs='+', required=True)
ap.add_argument('--out', type=str, default='data_mm')
ap.add_argument('--stride', type=int, default=5,
                help='Keep every Nth timestep. The window and the rollout both '
                     'run at 5 ms, so stride 5 stores exactly what training '
                     'reads and makes the file 5x smaller -- small enough to '
                     'sit entirely in page cache, which is what stops the disk '
                     'thrashing. Use 1 only if you want sub-5ms windows.')
ap.add_argument('--chunk', type=int, default=64,
                help='Scenarios read per HDF5 access. Larger = faster, more RAM '
                     '(33*8001*chunk*4 bytes each).')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

SCALARS = ['y_type', 'meta_alpha', 'meta_loc', 'y_t_fault', 'meta_t_clear',
           'y_ool', 'dp_status', 'failed', 'split', 't_apply', 't_remove',
           'meta_maxdev', 'base_maxdev_window', 'best_maxdev_window']

# ── pass 1: shapes ───────────────────────────────────────────────────────────
totals, T, C = [], None, None
for m in args.mat:
    with h5py.File(m, 'r') as f:
        c, t, n = f['X'].shape
        if T is None:
            T, C = t, c
        elif (t, c) != (T, C):
            raise SystemExit(f"{m}: shape ({c},{t}) != ({C},{T})")
        totals.append(n)
N = sum(totals)
T_out = len(range(0, T, args.stride))
print(f"{len(args.mat)} file(s), {N} scenarios x {T} steps x {C} channels")
print(f"stride {args.stride} -> {T_out} steps kept")
print(f"output: {N*T_out*C*4/2**30:.2f} GB float32")

mm = np.lib.format.open_memmap(os.path.join(args.out, 'X.npy'), mode='w+',
                               dtype=np.float32, shape=(N, T_out, C))

meta = {k: [] for k in SCALARS}
meta_extra = {}
off = 0
for m, n in zip(args.mat, totals):
    with h5py.File(m, 'r') as f:
        X = f['X']
        for k0 in range(0, n, args.chunk):
            k1 = min(k0 + args.chunk, n)
            blk = np.asarray(X[:, ::args.stride, k0:k1])   # (C, T_out, b)
            mm[off + k0:off + k1] = blk.transpose(2, 1, 0)
            done = off + k1
            print(f"\r  {done}/{N} ({100*done/N:.1f}%)", end='', flush=True)
        for k in SCALARS:
            if k in f:
                meta[k].append(np.array(f[k]).ravel())
        if not meta_extra:
            meta_extra['t'] = np.array(f['t']).ravel()[::args.stride]
            meta_extra['lines'] = np.array(f['lines']).T.astype(int)
            meta_extra['dt_solver'] = float(
                np.array(f['dt_solver']).ravel()[0]) * args.stride
            meta_extra['omega_limit'] = float(np.array(f['omega_limit']).ravel()[0])
            meta_extra['noise_std'] = np.array(f['noise_std']).ravel()
        if 'y_delta_p' in f:
            dp = np.array(f['y_delta_p'])
            meta_extra.setdefault('y_delta_p', []).append(
                dp.T if dp.shape[0] == 3 else dp)
    off += n
print()

mm.flush(); del mm

out = {k: np.concatenate(v) for k, v in meta.items() if v}
if 'y_delta_p' in meta_extra:
    out['y_delta_p'] = np.concatenate(meta_extra.pop('y_delta_p'))
out.update(meta_extra)
np.savez(os.path.join(args.out, 'meta.npz'), **out)

json.dump({'n_scenarios': int(N), 'n_steps': int(T_out), 'n_channels': int(C),
           'dt': meta_extra['dt_solver'], 'stride': args.stride,
           'files': args.mat},
          open(os.path.join(args.out, 'info.json'), 'w'), indent=1)
print(f"wrote {args.out}/X.npy and {args.out}/meta.npz")
