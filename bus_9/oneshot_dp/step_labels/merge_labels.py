"""Merge batches/batch_*.mat into step_labels.npz (k and step are 1-based, as
in MATLAB: k = scenario row of pinn_dataset.mat, step = index into t)."""
import glob, os
import numpy as np
import scipy.io as sio

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, 'batches', 'batch_*.mat')))
files = [f for f in files if not f.endswith('.tmp.mat')]
keys = ['k', 'step', 'dp', 'status', 'base_win', 'best_win']
acc = {k: [] for k in keys}; scen = []; secs = 0.0
for f in files:
    L = sio.loadmat(f)
    for k in keys:
        v = L[k]
        acc[k].append(v.reshape(-1, 3) if k == 'dp' else v.ravel())
    scen.append(L['scenarios'].ravel()); secs += float(L['seconds'].ravel()[0])
out = {k: np.concatenate(v) for k, v in acc.items()}
out['scenarios_done'] = np.concatenate(scen)
np.savez_compressed(os.path.join(here, 'step_labels.npz'), **out)
st = out['status']
print(f'{len(files)} batches, {len(out["scenarios_done"])} scenarios, {len(st)} fault steps, '
      f'{secs/3600:.1f} cpu-h; status counts', {int(s): int((st == s).sum()) for s in np.unique(st)})
