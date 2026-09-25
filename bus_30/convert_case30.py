"""
convert_case30.py
=================
One-time conversion of the sharded 30-bus dataset into a scenario-major memmap
plus a small metadata file.

    python convert_case30.py --labels case30_dataset.mat --shards case30_X_*.mat
    python convert_case30.py --mat case30_dataset_small.mat --out data30_small

Why a separate converter from the 9-bus one
-------------------------------------------
The 30-bus dataset is split: `case30_dataset.mat` holds the labels for all
10000 scenarios, and `X` lives in 40 shard files of 250 scenarios each. Each
shard carries `scenario_ids` (1-based) saying which rows it holds, so shards
can arrive in any order and are placed by id, not by position.

Layout and dtype follow the 9-bus converter for a reason:

  * `X` in the .mat is (C, T, n) with the scenario index on the FASTEST axis,
    so reading one scenario's window touches 90*8001 locations, each n*4 bytes
    apart. The output is (N, T_out, 90) in C order, so a window read is one
    contiguous strided slice.
  * float32, not float16: omega sits at ~1.0 with deviations of 0.002-0.02 and
    the limit is 0.005, while float16 spacing near 1.0 is ~0.001 -- it would
    destroy the one signal everything depends on.
  * reading is TIME-MAJOR. The file is chunked so that every scenario lives
    inside every chunk; looping over scenario blocks would decompress all
    28.8 GB once per block. Looping over time touches each chunk once, and
    --agg decimate skips (stride-1)/stride of them entirely.

Sampling
--------
The 9-bus dataset stored every 10th ms; this one stores every ms, so its omega
is much noisier sample to sample at the same noise_std. `--stride 10 --agg
decimate` (the default) reproduces 9-bus conditions exactly: same 10 ms spacing,
same 5e-5 noise floor, so the two systems stay comparable. `--agg mean` instead
averages each group of 10, which drops the noise to 1.6e-5 and is what a real
PMU does -- better SNR, but it must read the whole 28.8 GB and it moves the
noise floor, so the trajectory-loss target is no longer the 9-bus one.

Output (28.8 GB -> 2.88 GB at stride 10):
    <out>/X.npy      (N, T_out, 90) float32
    <out>/meta.npz   labels, metadata, time vector
    <out>/info.json  sizes, dt, stride
"""

import argparse, glob, json, os

import h5py
import numpy as np
import scipy.io as sio

ap = argparse.ArgumentParser()
ap.add_argument('--labels', default='case30_dataset.mat',
                help='file holding the labels for all scenarios')
ap.add_argument('--shards', nargs='*', default=None,
                help='shard files holding X (default: case30_X_*.mat next to '
                     '--labels). Order does not matter; scenario_ids decide '
                     'placement.')
ap.add_argument('--mat', default=None,
                help='single file holding BOTH X and the labels (the small '
                     'dataset). Overrides --labels/--shards.')
ap.add_argument('--dp', default=None,
                help='optional separate delta_p label file, if the labels file '
                     'does not already carry y_delta_p/dp_status.')
ap.add_argument('--out', default='data30_mm')
ap.add_argument('--stride', type=int, default=10,
                help='keep every Nth timestep (default 10 = 10 ms, matching '
                     'the 9-bus dataset)')
ap.add_argument('--agg', choices=['decimate', 'mean'], default='decimate')
ap.add_argument('--tblock', type=int, default=20,
                help='output timesteps buffered per write; RAM is '
                     'n*tblock*C*4 bytes per shard')
args = ap.parse_args()

SCALARS = ['y_type', 'meta_alpha', 'meta_loc', 'y_t_fault', 'meta_t_clear',
           'y_ool', 'dp_status', 'failed', 'split', 't_apply', 't_remove',
           'meta_maxdev', 'base_maxdev_window', 'best_maxdev_window',
           'base_maxdev_all', 'verify_maxdev']

# ── resolve inputs ───────────────────────────────────────────────────────────
if args.mat:
    label_file, shards = args.mat, [args.mat]
else:
    label_file = args.labels
    shards = args.shards
    if not shards:
        shards = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(label_file)), 'case30_X_*.mat')))
    if not shards:
        raise SystemExit("no shard files found; pass --shards or --mat")
for f_ in [label_file] + list(shards):
    if not os.path.exists(f_):
        raise SystemExit(f"not found: {f_}")

os.makedirs(args.out, exist_ok=True)


def read_any(f, k):
    """Read a variable from an h5py file (v7.3) as a flat/plain array."""
    return np.array(f[k])


# ── labels ───────────────────────────────────────────────────────────────────
meta = {}
with h5py.File(label_file, 'r') as f:
    for k in SCALARS:
        if k in f:
            meta[k] = read_any(f, k).ravel()
    t_all = read_any(f, 't').ravel()
    T = len(t_all)
    if 'y_delta_p' in f:
        dp = read_any(f, 'y_delta_p')
        meta['y_delta_p'] = dp.T if dp.shape[0] != len(meta['y_type']) else dp
    br = read_any(f, 'branches').astype(int)
    meta['branches'] = br.T if br.shape[0] == 2 else br
    for k in ('n_machines', 'n_buses'):
        if k in f:
            meta[k] = int(read_any(f, k).ravel()[0])
    dt_solver = float(read_any(f, 'dt_solver').ravel()[0])
    meta['omega_limit'] = float(read_any(f, 'omega_limit').ravel()[0])
    noise = read_any(f, 'noise_std').ravel()
    if 'gen_bus' in f:
        meta['gen_bus'] = read_any(f, 'gen_bus').ravel().astype(int)

N = len(meta['y_type'])

# delta_p labels can live in their own file
if args.dp or 'y_delta_p' not in meta:
    dp_file = args.dp or os.path.join(os.path.dirname(os.path.abspath(label_file)),
                                      'case30_delta_p_labels.mat')
    if os.path.exists(dp_file):
        try:
            d = sio.loadmat(dp_file, squeeze_me=True)
        except NotImplementedError:                      # v7.3
            with h5py.File(dp_file, 'r') as fd:
                d = {k: np.array(fd[k]) for k in fd if not k.startswith('#')}
        for k in ('y_delta_p', 'dp_status', 't_apply', 't_remove',
                  'base_maxdev_all', 'base_maxdev_window',
                  'best_maxdev_window', 'verify_maxdev'):
            if k in d:
                a = np.asarray(d[k])
                if k == 'y_delta_p':
                    meta[k] = a.T if a.shape[0] != N else a
                else:
                    meta[k] = a.ravel()
        print(f"delta_p labels read from {os.path.basename(dp_file)}")
    elif 'y_delta_p' not in meta:
        print("WARNING: no delta_p labels found -- training needs them")

T_out = len(range(0, T, args.stride))

# ── pass 1: shard shapes and ids ─────────────────────────────────────────────
ids_of, C = {}, None
for m in shards:
    with h5py.File(m, 'r') as f:
        c, t, n = f['X'].shape
        if C is None:
            C = c
        elif c != C:
            raise SystemExit(f"{m}: {c} channels, expected {C}")
        if t != T:
            raise SystemExit(f"{m}: {t} timesteps, labels say {T}")
        if 'scenario_ids' in f:
            ids = np.array(f['scenario_ids']).ravel().astype(int) - 1
        else:
            ids = np.arange(n)            # single-file small dataset
        if len(ids) != n:
            raise SystemExit(f"{m}: {len(ids)} ids for {n} scenarios")
        ids_of[m] = ids

covered = np.concatenate(list(ids_of.values()))
if len(np.unique(covered)) != len(covered):
    raise SystemExit("shards overlap: a scenario id appears twice")
n_cov = len(covered)
print(f"{len(shards)} shard(s), {n_cov} of {N} scenarios, "
      f"{T} steps x {C} channels")
print(f"stride {args.stride} ({args.agg}) -> {T_out} steps kept")
print(f"output: {n_cov * T_out * C * 4 / 2**30:.2f} GB float32")
if n_cov < N:
    print(f"NOTE: {N - n_cov} scenarios have no shard; their rows stay zero. "
          f"Pass every shard for a full conversion.")

mm = np.lib.format.open_memmap(os.path.join(args.out, 'X.npy'), mode='w+',
                               dtype=np.float32, shape=(N, T_out, C))

# ── pass 2: copy, time-major ─────────────────────────────────────────────────
for si, m in enumerate(shards):
    ids = ids_of[m]
    n = len(ids)
    order = np.argsort(ids)               # memmap writes in increasing id
    ids_sorted = ids[order]
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
                if full < ng:
                    parts.append(raw[:, full:, :].mean(1, keepdims=True))
                g = np.concatenate(parts, axis=1)
            else:
                g = np.empty((C, j1 - j0, n), np.float32)
                for jj, t_ in enumerate(range(j0 * args.stride,
                                              j1 * args.stride, args.stride)):
                    g[:, jj, :] = X[:, t_, :]
            block = g.transpose(2, 1, 0)[order]          # (n, j1-j0, C)
            mm[ids_sorted, j0:j1, :] = block
            print(f"\r  [{si+1}/{len(shards)}] {os.path.basename(m)}: "
                  f"{j1}/{T_out} steps ({100*j1/T_out:5.1f}%)",
                  end='', flush=True)
    print()

mm.flush(); del mm

# ── metadata ─────────────────────────────────────────────────────────────────
if args.agg == 'mean':
    meta['t'] = np.array([t_all[j * args.stride:min((j + 1) * args.stride, T)].mean()
                          for j in range(T_out)])
    meta['noise_std'] = noise / np.sqrt(args.stride)
else:
    meta['t'] = t_all[::args.stride]
    meta['noise_std'] = noise
meta['dt_solver'] = dt_solver * args.stride
meta['agg'] = args.agg
meta['scenario_covered'] = np.isin(np.arange(N), covered)

np.savez(os.path.join(args.out, 'meta.npz'), **meta)
json.dump({'n_scenarios': int(N), 'n_covered': int(n_cov),
           'n_steps': int(T_out), 'n_channels': int(C),
           'n_machines': int(meta.get('n_machines', 6)),
           'n_buses': int(meta.get('n_buses', 30)),
           'dt': float(meta['dt_solver']), 'stride': args.stride,
           'agg': args.agg, 'system': 'case30',
           'labels': label_file, 'shards': list(shards)},
          open(os.path.join(args.out, 'info.json'), 'w'), indent=1)
print(f"wrote {args.out}/X.npy and {args.out}/meta.npz")
