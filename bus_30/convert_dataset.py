"""
convert_dataset.py
==================
One-time conversion of pinn_dataset.mat into a scenario-major memmap plus a
small metadata file.

    python convert_dataset.py --mat pinn_dataset.mat --out data_mm

Why this is needed
------------------
Works for both systems: the 9-bus (33 channels, 3 machines, 9 buses) and the
30-bus (90 channels, 6 machines, 30 buses). Sizes are read from the file.

The .mat stores X as (C, 8001, N) with the scenario index on the FASTEST
varying axis, so reading one scenario's window touches 33*8001 locations each
N*4 bytes apart -- the worst possible access pattern, and it gets worse as the
dataset grows. The converter rewrites it as (N, 8001, 33) in C order, so one
scenario's time-block is contiguous and a window read is a single strided slice
of a few hundred KB.

float32 is kept deliberately. float16 would halve the file, but omega sits at
~1.0 with deviations of 0.002-0.02 and the out-of-limit threshold is 0.005;
float16 spacing near 1.0 is about 0.001, which would corrupt the single most
important signal in the dataset. The 30-bus file is 28.8 GB at stride 1 and
5.8 GB at stride 5, which still fits page cache on a 32 GB machine; the 28.8 GB
is streamed through in --chunk-sized blocks and never held whole.

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
ap.add_argument('--agg', choices=['decimate', 'mean'], default='decimate',
                help="How to shrink the time axis. 'decimate' keeps every Nth "
                     "sample and reads only every Nth HDF5 chunk, so it touches "
                     "1/stride of the file. 'mean' averages each group of "
                     "`stride` samples: noise drops by sqrt(stride) (5e-5 -> "
                     "1.6e-5 at stride 10) and it is what a real PMU does, but "
                     "it must read the WHOLE file. Times are group centres "
                     "under 'mean', first-sample times under 'decimate'.")
ap.add_argument('--tblock', type=int, default=20,
                help='Output timesteps buffered per write. RAM is '
                     'N*tblock*C*4 bytes (10000 x 20 x 90 ~ 720 MB).')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

SCALARS = ['y_type', 'meta_alpha', 'meta_loc', 'y_t_fault', 'meta_t_clear',
           'y_ool', 'dp_status', 'failed', 'split', 't_apply', 't_remove',
           'meta_maxdev', 'base_maxdev_window', 'best_maxdev_window',
           'base_maxdev_all', 'verify_maxdev']

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
NG_guess = (C - 2 * 30) // 5 if C == 90 else (C - 2 * 9) // 5
print(f"{len(args.mat)} file(s), {N} scenarios x {T} steps x {C} channels")
print(f"stride {args.stride} -> {T_out} steps kept")
print(f"output: {N*T_out*C*4/2**30:.2f} GB float32")

mm = np.lib.format.open_memmap(os.path.join(args.out, 'X.npy'), mode='w+',
                               dtype=np.float32, shape=(N, T_out, C))

meta = {k: [] for k in SCALARS}
meta_extra = {}
off = 0
# Read TIME-MAJOR, not scenario-major.
#
# Both datasets are gzip-chunked as (C, few, N) -- every scenario lives inside
# every chunk. Reading a block of scenarios therefore has to decompress every
# chunk it spans, so looping over scenario blocks decompresses the entire file
# once PER BLOCK: ~20x redundant work on the 9-bus file and unaffordable on the
# 28.8 GB 30-bus one. Looping over time touches each chunk exactly once, and
# with --agg decimate it skips (stride-1)/stride of the chunks entirely.
for m, n in zip(args.mat, totals):
    with h5py.File(m, 'r') as f:
        X = f['X']
        for j0 in range(0, T_out, args.tblock):
            j1 = min(j0 + args.tblock, T_out)
            if args.agg == 'mean':
                t0, t1 = j0 * args.stride, min(j1 * args.stride, T)
                raw = np.asarray(X[:, t0:t1, :], dtype=np.float32)
                ng = t1 - t0
                full = (ng // args.stride) * args.stride
                parts = [raw[:, :full, :].reshape(C, -1, args.stride, n).mean(2)]
                if full < ng:                      # ragged final group
                    parts.append(raw[:, full:, :].mean(1, keepdims=True))
                g = np.concatenate(parts, axis=1)
            else:
                g = np.empty((C, j1 - j0, n), np.float32)
                for jj, t_ in enumerate(range(j0 * args.stride,
                                              j1 * args.stride, args.stride)):
                    g[:, jj, :] = X[:, t_, :]
            mm[off:off + n, j0:j1, :] = g.transpose(2, 1, 0)
            print(f"\r  {m}: {j1}/{T_out} steps ({100*j1/T_out:.1f}%)",
                  end='', flush=True)
        for k in SCALARS:
            if k in f:
                meta[k].append(np.array(f[k]).ravel())
        if not meta_extra:
            t_all = np.array(f['t']).ravel()
            if args.agg == 'mean':                 # group centres
                gi = [t_all[j*args.stride:min((j+1)*args.stride, T)].mean()
                      for j in range(T_out)]
                meta_extra['t'] = np.array(gi)
            else:
                meta_extra['t'] = t_all[::args.stride]
            bkey = 'branches' if 'branches' in f else 'lines'
            br = np.array(f[bkey]).astype(int)
            meta_extra['branches'] = br.T if br.shape[0] == 2 else br
            for k in ('n_machines', 'n_buses'):
                if k in f:
                    meta_extra[k] = int(np.array(f[k]).ravel()[0])
            meta_extra['dt_solver'] = float(
                np.array(f['dt_solver']).ravel()[0]) * args.stride
            meta_extra['omega_limit'] = float(np.array(f['omega_limit']).ravel()[0])
            ns = np.array(f['noise_std']).ravel()
            # averaging reduces the effective measurement noise
            meta_extra['noise_std'] = ns / np.sqrt(args.stride) if args.agg == 'mean' else ns
            meta_extra['agg'] = args.agg
        if 'y_delta_p' in f:
            dp = np.array(f['y_delta_p'])
            meta_extra.setdefault('y_delta_p', []).append(
                dp.T if dp.shape[1] == n else dp)
    off += n
print()

mm.flush(); del mm

out = {k: np.concatenate(v) for k, v in meta.items() if v}
if 'y_delta_p' in meta_extra:
    out['y_delta_p'] = np.concatenate(meta_extra.pop('y_delta_p'))
out.update(meta_extra)
np.savez(os.path.join(args.out, 'meta.npz'), **out)

json.dump({'n_scenarios': int(N), 'n_steps': int(T_out), 'n_channels': int(C),
           'n_machines': int(meta_extra.get('n_machines', NG_guess)),
           'n_buses': int(meta_extra.get('n_buses', (C - 5 * NG_guess) // 2)),
           'dt': meta_extra['dt_solver'], 'stride': args.stride,
           'agg': args.agg,
           'files': args.mat},
          open(os.path.join(args.out, 'info.json'), 'w'), indent=1)
print(f"wrote {args.out}/X.npy and {args.out}/meta.npz")
