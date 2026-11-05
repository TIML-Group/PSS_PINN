"""
koopman_baseline.py — consolidated, committed one-script entry for the
9-bus Koopman-MPC baseline (Korda, Susuki & Mezić 2018, arXiv:1803.10744).

Provenance
----------
- CONSOLIDATES the gitignored scratch scripts (bus_9/_koopa_edmd.py,
  _koopa_edmd_angdiff.py, _koopa_edmd_angdiff_deg3.py, _koopa_edmd_fullchan.py,
  _koopa_horizon_curve.py, _koopa_mpc.py, _koopa_sweep.py + the probe helpers
  _koopa_merge_scan.py / _koopa_fix_scan_id.py / _koopa_probe_design.py) into
  one flag-driven script. The probe/dead variants (raw-delta deg-2/deg-3 fits,
  horizon curve, design probes) are NOT shipped here — their full JSON record
  stays in bus_9/_koopman_out/ and tasks12_log.md.
- Baseline: Korda/Susuki/Mezić 2018 "Linear Prediction Model Control for
  Inconsistent Linearized Systems" (arXiv:1803.10744) — published, problem-
  matched (transient stabilization of swing dynamics), channel-matched
  (control = fraction of nominal mechanical power ≈ governor ΔP), no training
  (one EDMD linear least-squares).
- Honest-baseline rules (fixed in the plan BEFORE any code, kept here):
  * no encoder, no fault knowledge, no schedule — the baseline sees the same
    noisy measured window the encoder sees; the asymmetry is the finding;
  * EDMD is fitted on pre-fault+governor dynamics only, hypothesis-blind,
    exactly Korda's design; no tuning of the baseline toward the oracle;
  * the baseline's own knobs (dictionary degree, horizon H, QP weights) are
    tuned and disclosed without label leakage.
  Wiki record: 'PINN wiki/queries/koopman-mpc-baseline-plan.md' (shipped
  configurations, findings, oracle caveat) + the wiki log.md.

Shipped configuration (ALL numbers frozen in bus_9/_koopman_out/*.json)
----------------------------------------------------------------------
- H = 20 model steps @ dt_true = 5 ms (100 ms horizon LIMITED baseline);
  decision clock 20 ms (every 4th true step), first cell's command held.
- QP knobs: w=1.0 (hinge on predicted |omega-1|, deadband 5e-4), terminal
  sync Wt=10 (no deadband), effort qu=1e-4, |u| <= 1 pu per machine,
  rate |u_c - u_{c-1}| <= 0.5 per 20 ms decision. osqp, eps 1e-6, warm
  (bounds update + solve) and cold (fresh setup + solve) both logged.
- Steady lifted Kalman filter: physical-state form (prediction =
  phys(A phi(xhat) + B u), re-lifted every 5 ms step), one DARE on the
  pre-fault-equilibrium linearization F, C = measured noisy channels
  (delta, omega, Pm), R = dataset noise stds, Q = q*I with q = 1e-4 sized
  NON-LEAKILY from pre-fault blind-observer u-rail tracking.
- Fits (--fit): fullchan  (33-channel representation, n_phi = 258,
  task2_edmd_fullchan.json) · angdiff2 (angle-difference deg-2, n_phi = 94,
  task2_edmd_angdiff.json) · bare (bare-swing Korda-native refit, n_phi = 58,
  task34_bare_refit.json). All: seed 42, 4000 episodes x 200 steps @ 5 ms,
  same episodic x0/u mixtures and draw order, 90/10 split BY EPISODE,
  216k/24k tuples, fresh-u mixture per tuple, one lstsq, constant row pinned.
- Sweep protocol (Task 6): seed-42 shared draw (compare_out_s42/instants.json,
  split test), true subset = first 2048 instants (compare_models --n_true
  convention, 484 scenarios), dp_hold='clear', grace 0.2 s, omega_limit 0.005,
  rescue/HARMED metrics with 95% cluster-bootstrap CIs (B=2000, rng 49,
  compare_models protocol), head rows CITED from canonical seed-42 artifacts
  (never re-run).

ORACLE-ROW CAVEAT (read before using the sweep's ORACLE numbers)
----------------------------------------------------------------
The sweep's "oracle_label" row re-emits the label ΔP at the DECISION instant
tk — tk-blind, deliberately mis-timed relative to the label's own design
schedule t_apply = t_fault + 0.05. It is NOT the ceiling: labels applied at
their own design schedule give rescue 0.991 (validated against the stored
best_maxdev_window). In the paper cite 0.991 (design-timing) as the oracle
ceiling and use the tk-blind row only as the "fixed schedule,
decision-triggered" contrast — never present it as "the optimum".

Subcommands
-----------
  smoke       Task 1 — clean-dynamics sampler check vs stored data (gate 5e-5)
  fit-edmd    Task 2 — EDMD lifted linear predictor, --fit {fullchan,angdiff2,bare}
  fit-kf      Task 3 — steady lifted KF + fault-time estimation record
  run-mpc     Task 4 — dense-form QP MPC, --channel {gov_fullchan,korda_native_bare}
  sweep       Task 6 — seed-42 validation table (--row, --n subset; writes
              baseline_table[.json or _n<k>.json] + quality/latency figures)
  scan        Stretch — scheduled-scan row (ONE attempt, fixed recipe; full draw)
  crosscheck  Task 5 — scorer vs verify_true.py itself (GATE: exact residuals)

Each subcommand writes its JSON into --out (default bus_9/_koopman_out) and
appends nothing to git-tracked state. Fit/KF/fit sources are read from --out,
falling back to the shipped default out dir so verification runs can reuse
the shipped artifacts with a scratch --out.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.linalg as sla
import scipy.sparse as sparse

from ieee9_physics import IEEE9, HYPOTHESES

BUS9 = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BUS9, 'data_mm')
DEFAULT_OUT = os.path.join(BUS9, '_koopman_out')
INSTANTS_DEFAULT = os.path.join(BUS9, 'compare_out_s42', 'instants.json')
COMP = os.path.join(BUS9, 'compare_out_s42')

# dataset channel indices (33 channels, DATASET.md / verify_true.py)
D_IDX = [0, 2, 4]
W_IDX = [1, 3, 5]
PM_IDX = [6, 7, 8]
VM_IDX = list(range(15, 33, 2))
TH_IDX = list(range(16, 33, 2))
TYPE_NAMES = ['gen', 'load', 'line']   # y_type is 1-based: 1=gen 2=load 3=line

DT = DT_TRUE = 0.005
DEC_EVERY = 4                 # controller decision every 4 true steps = 20 ms
H = 20                        # prediction horizon: 20 x 5 ms = 100 ms
N_DEC = H // DEC_EVERY        # 5 control cells

NOISE_STD = np.array([.01, .01, .01, 5e-5, 5e-5, 5e-5, 5e-3, 5e-3, 5e-3])
KF_Q = 1e-4                   # shipped KF knob (sized pre-fault; see docstring)
QP_W = 1.0                    # hinge weight per step/machine
QP_DB = 5e-4                  # action deadband on predicted |omega-1|
QP_WT = 10.0                  # terminal sync weight (deadband 0)
QP_QU = 1e-4                  # effort quadratic weight (grid-probed, disclosed)
QP_UMAX = 1.0                 # |u| <= 1 pu per machine
QP_RATE = 0.5                 # |u_c - u_{c-1}| <= 0.5 per 20 ms decision

SCEN_ORDER = [902, 4723, 6978]   # 902 gen-trip, 4723 load-step = gate pair;

# sweep / scoring conventions (frozen in the shipped artifacts)
GRACE = 0.2                     # RolloutController default (verify_true grace)
N_TRUE = 2048                   # compare_models --n_true convention
AGE_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)]
AGE_LABELS = ['0-50ms', '50-100ms', '100-200ms', '200-500ms', '>500ms']
BOOT = 2000
BOOT_SEED = 49                  # compare_models: args.seed(42) + 7
OMEGA_LIMIT = 0.005             # md['omega_limit'] (asserted at runtime)
CHUNK = 256                     # instants per floor/oracle chunk (memory-bound)
COLD_SC = 7                     # cold-timing sampling: every 7th scenario
COLD_DEC = 11                   # x every 11th decision

# scheduled-scan stretch (ONE attempt, fixed recipe)
SCAN_CHUNK = 24                 # instants per chunk -> 24 x 77 = 1848 rows/chunk
SCAN_DIR_NAMES = ['+e1', '+e2', '+e3', '-e1', '-e2', '-e3', 'avg3', 'avneg3',
                  'd1-d2', 'd1-d3']
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
SCAN_SCALE_MAX = 1.0            # same |u| rail as the koopman rows


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def resolve_out(args):
    """Output/scratch --out dir; artifacts created under it."""
    out = os.path.join(BUS9, args.out) if not os.path.isabs(args.out) \
        else args.out
    os.makedirs(out, exist_ok=True)
    return out


def find_artifact(name, out_dir):
    """Fit/source JSON lookup: --out first, then the shipped default out dir."""
    for cand in (os.path.join(out_dir, name),
                 os.path.join(DEFAULT_OUT, name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"required artifact {name} not found in {out_dir} or {DEFAULT_OUT} — "
        f"run the corresponding fit subcommand first")


def append_log(out_dir, cmd):
    with open(os.path.join(out_dir, 'tasks12_log.md'), 'a', encoding='utf-8') \
            as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} `{cmd}`\n"
                f"- see JSON artifact in this dir for full numbers\n")


# ══════════════════════════════════════════════════════════════════════════════
# dataset / scenario helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir=DATA):
    X = np.load(os.path.join(data_dir, 'X.npy'), mmap_mode='r')
    md = np.load(os.path.join(data_dir, 'meta.npz'))
    return dict(
        X=X,
        split=md['split'].ravel().astype(int),
        y_type=md['y_type'].ravel().astype(int),
        meta_loc=md['meta_loc'].ravel().astype(int),
        meta_alpha=md['meta_alpha'].ravel(),
        t_fault=md['y_t_fault'].ravel(),
        t_clear=md['meta_t_clear'].ravel(),
        t_grid=md['t'].ravel(),
        lines=md['lines'].astype(int),
    )


def select_scenarios(d, n_per_type, seed):
    """Validation (split==2) scenarios, n_per_type per fault type, seeded."""
    rng = np.random.default_rng(seed)
    idx = []
    for t in (1, 2, 3):
        cand = np.sort(np.where((d['split'] == 2) & (d['y_type'] == t))[0])
        idx += list(rng.choice(cand, size=min(n_per_type, len(cand)),
                               replace=False))
    return np.sort(np.asarray(idx, dtype=int))


def scenario_hyp(S, d, k):
    return HYPOTHESES.index((TYPE_NAMES[d['y_type'][k] - 1],
                             int(d['meta_loc'][k])))


def load_instants(instants_file):
    inst = json.load(open(instants_file))
    assert inst['seed'] == 42 and inst['split'] == 'test'
    return np.asarray(inst['ks'], dtype=int), np.asarray(inst['tk'],
                                                         dtype=float), inst


def ckpt_args(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck['args'], ck


def build_windows(A, d, KS, TK):
    """verify_true.make_batch convention: valid_t = clip(t_change - tk, 0, horizon)."""
    tf, tc = d['t_fault'][KS], d['t_clear'][KS]
    active = (TK >= tf) & (TK < tc)
    t_change = np.where(active, tc, np.where(TK < tf, tf, np.inf))
    vt = np.clip(t_change - TK, 0.0, float(A['horizon']))
    return active, vt


def import_verify_true(ckpt, out_dir):
    """Import verify_true.py with an argv stub. Module-level argparse: run its
    --n 1 main entry once (cheap) so its module globals (model, S, make_batch,
    full_sim, peaks_from) are the LITERAL reference machinery. Its outputs go
    to a scratch dir under the --out dir (no existing file touched)."""
    ref_dir = os.path.join(out_dir, '_crosscheck_s42', 'verify_refimport')
    sys.argv = ['verify_true.py', '--ckpt', ckpt, '--data', 'data_mm', '--n',
                '1', '--split', 'test', '--instants-file', INSTANTS_DEFAULT,
                '--out', ref_dir, '--plots_per_type', '0']
    import verify_true as vtm                    # noqa   (runs the module)
    return vtm


def run_verify_subprocess(out_dir, n, split='test',
                          ckpt='receding_cascade/refined/best.pt'):
    """The OFFICIAL verify_true.py run on the same shared instants."""
    sub_dir = os.path.join(out_dir, '_crosscheck_s42', 'verify_sub')
    cmd = [os.path.join(BUS9, '..', '.venv', 'Scripts', 'python.exe'),
           'verify_true.py', '--ckpt', ckpt,
           '--data', 'data_mm', '--n', str(n), '--batch', str(max(n, 1)),
           '--split', split, '--instants-file', INSTANTS_DEFAULT,
           '--out', sub_dir]
    print('subprocess:', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=BUS9, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise RuntimeError('verify_true.py subprocess failed')
    print('  done:', r.stdout.strip().splitlines()[-2] if r.stdout else '')
    return json.load(open(os.path.join(sub_dir, 'verify_results.json')))


def load_meta():
    return np.load(os.path.join(DATA, 'meta.npz'), allow_pickle=True)


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — clean-dynamics reproduction (smoke)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_fault_networks(S, d, ks):
    """Exact faulted network + set-point offset from the METADATA (not the
    model's hypothesis), same pattern as verify_true.true_network."""
    n = len(ks)
    Yf = torch.zeros(n, 3, 3, dtype=S.cdtype)
    dPm = torch.zeros(n, 3, dtype=S.rdtype)
    for j, k in enumerate(ks):
        h = scenario_hyp(S, d, k)
        a = torch.tensor([d['meta_alpha'][k]], dtype=S.rdtype)
        Yf[j], _ = S.kron_reduce(S.faulted_Y(h, a))
        dPm[j] = S.dPm0(h, a)[0]
    return Yf, dPm


@torch.no_grad()
def simulate_clean(S, d, ks, dt=DT_TRUE, dP_extra=None):
    """Batched 8 s clean simulation: pre-fault equilibrium -> faulted Yred on
    [t_fault, t_clear) -> pre-fault. Returns full states (n, T, 9) numpy."""
    n = len(ks)
    Yf, dPm = build_fault_networks(S, d, ks)
    Ypre = S.Yred_pre.expand(n, 3, 3)
    Mp, Np = S.pe_consts(Ypre)
    Mf, Nf = S.pe_consts(Yf)
    tf = torch.tensor(d['t_fault'][ks], dtype=S.rdtype)
    tc = torch.tensor(d['t_clear'][ks], dtype=S.rdtype)
    x = torch.cat([S.delta0, torch.ones(3), S.Pm0]) \
        .unsqueeze(0).expand(n, 9).clone()
    n_steps = int(round(float(d['t_grid'][-1]) / dt))
    states = torch.zeros(n, n_steps + 1, 9, dtype=S.rdtype)
    states[:, 0] = x
    Ps_extra = None
    if dP_extra is not None:
        Ps_extra = torch.tensor(dP_extra, dtype=S.rdtype)
    for i in range(n_steps):
        tm = i * dt + dt / 2
        act = ((tm >= tf) & (tm < tc)).view(n, 1)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.float()
        if Ps_extra is not None:
            Ps = Ps + Ps_extra
        x = S.rk4_real(x, M, N, Ps, dt)
        states[:, i + 1] = x
    return states.numpy()


@torch.no_grad()
def cmd_smoke(args, out_dir):
    t0 = time.time()
    d = load_data()
    ks = select_scenarios(d, args.n_per_type, args.seed)
    print("scenarios:", [(int(k), TYPE_NAMES[d['y_type'][k] - 1],
                          float(d['meta_alpha'][k]), int(d['meta_loc'][k]),
                          float(d['t_fault'][k]), float(d['t_clear'][k]))
                         for k in ks])

    S = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
              os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
              dtype=torch.float64, lines=d['lines'])
    dY = (S.Yred_pre - S.Yred_pre_ref).abs().max().item()
    dV = (S.Vrec_pre - S.Vrec_pre_ref).abs().max().item()
    print(f"Yred_pre vs stored ref max|diff|: {dY:.3e}   Vrec_pre: {dV:.3e}")

    sim = simulate_clean(S, d, ks)          # (n, T, 9) float64
    dt_store = float(d['t_grid'][1] - d['t_grid'][0])
    assert abs(dt_store - DT_TRUE) < 1e-9, (dt_store, DT_TRUE)

    scen = []
    om_all, peak_err_all, om_seg = [], [], {'pre': [], 'fault': [], 'post': []}
    for j, k in enumerate(ks):
        st = np.asarray(d['X'][k], dtype=np.float64)
        clean_w = sim[j][:, 3:6]
        clean_d = sim[j][:, 0:3]
        clean_pm = sim[j][:, 6:9]
        stw, std_, stp = st[:, W_IDX], st[:, D_IDX], st[:, PM_IDX]
        err = clean_w - stw
        seg = {'pre': d['t_grid'] < d['t_fault'][k] - 1e-9,
               'fault': (d['t_grid'] >= d['t_fault'][k] - 1e-9)
                        & (d['t_grid'] < d['t_clear'][k] - 1e-9),
               'post': d['t_grid'] >= d['t_clear'][k] - 1e-9}
        rmse = {name: float(np.sqrt(np.mean(err[m] ** 2)))
                for name, m in seg.items()}
        rmse['overall'] = float(np.sqrt(np.mean(err ** 2)))
        peak_clean = float(np.abs(clean_w - 1.0).max())
        peak_stored = float(np.abs(stw - 1.0).max())
        scen.append({
            'scenario': int(k), 'type': TYPE_NAMES[d['y_type'][k] - 1],
            'alpha': float(d['meta_alpha'][k]), 'loc': int(d['meta_loc'][k]),
            't_fault': float(d['t_fault'][k]), 't_clear': float(d['t_clear'][k]),
            'rmse_omega': rmse,
            'rmse_omega_max_seg': max(rmse[s] for s in ('pre', 'fault', 'post')),
            'rmse_delta': float(np.sqrt(np.mean((clean_d - std_) ** 2))),
            'rmse_Pm': float(np.sqrt(np.mean((clean_pm - stp) ** 2))),
            'peak_dev_clean': peak_clean, 'peak_dev_stored': peak_stored,
        })
        om_all.append(rmse['overall'])
        peak_err_all.append(abs(peak_clean - peak_stored))
        for name, m in seg.items():
            om_seg[name].append(rmse[name])
        print(f"  k={k} {scen[-1]['type']:4s} a={d['meta_alpha'][k]:.2f} "
              f"omega RMSE {rmse['overall']:.3e} "
              f"(pre {rmse['pre']:.2e} filt {rmse['fault']:.2e} "
              f"post {rmse['post']:.2e})  "
              f"peak {peak_clean:.4f} vs {peak_stored:.4f}")

    gate = {
        'omega_rmse_max_overall': float(max(om_all)),
        'omega_rmse_max_segment': float(max(max(s) for s in om_seg.values())),
        'peak_err_max': float(max(peak_err_all)),
        'omega_floor_reference_5e5': 5.008e-5,
        'pass_omega_rmse_within_5e4_of_floor': bool(max(om_all) <= 5e-4),
        'pass_peak_within_5e4': bool(max(peak_err_all) <= 5e-4),
        'PASS': bool(max(om_all) <= 5e-4 and max(peak_err_all) <= 5e-4),
    }
    out = {'cmd': ' '.join(sys.argv), 'seed': args.seed,
           'runtime_s': time.time() - t0,
           'Yred_pre_diff': dY, 'Vrec_pre_diff': dV,
           'noise_std': {'omega': 5e-5, 'delta': 0.01, 'Pm': 5e-3},
           'scenarios': scen, 'gate': gate}
    path = os.path.join(out_dir, 'task1_smoke.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('gate:', json.dumps(gate, indent=1))
    print(f"saved {path}  ({time.time() - t0:.1f} s)")


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — EDMD fits (shared seed-42 protocol, three shipped dictionary
# variants; the raw-delta deg-2/deg-3 fits stay as their JSON record only)
# ══════════════════════════════════════════════════════════════════════════════

def mono_indices(nvar, degree):
    """Index tuples of exactly-degree-`degree` monomials with repetition."""
    return list(itertools.combinations_with_replacement(range(nvar), degree))


def _pair_trig(dlt, pairs):
    cols = []
    for (i, j) in pairs:
        dd = dlt[:, i][:, None] - dlt[:, j][:, None]
        cols.append(np.cos(dd))
        cols.append(np.sin(dd))
    return np.concatenate(cols, axis=1)


# ── angdiff2 (94-dim approved angle-difference dictionary) ────────────────────
N_PHI_ANGDIFF = 94
DICT_SPEC_ANGDIFF = ('phi = [1, delta(3) linear-observable-only, '
                     'd21=d2-d1, d31=d3-d1, omega(3), Pm(3), '
                     'cos(d21), sin(d21), cos(d31), sin(d31), '
                     'degree-2 monomials of {d21, d31, omega, Pm, cos/sin '
                     'terms} (no raw delta in any monomial)]')


def phi_angdiff(X):
    """X: (N,9) = [delta(3), omega(3), Pm(3)] -> (N,94)."""
    dlt, w, pm = X[:, 0:3], X[:, 3:6], X[:, 6:9]
    d21 = (dlt[:, 1] - dlt[:, 0])[:, None]
    d31 = (dlt[:, 2] - dlt[:, 0])[:, None]
    trig = np.concatenate([np.cos(d21), np.sin(d21),
                           np.cos(d31), np.sin(d31)], axis=1)
    v12 = np.concatenate([d21, d31, w, pm, trig], axis=1)      # (N,12)
    mon = np.stack([np.prod(v12[:, c], axis=1)
                    for c in mono_indices(12, 2)], axis=1)     # (N,78)
    return np.concatenate([np.ones((X.shape[0], 1)), dlt, v12, mon], axis=1)


# ── fullchan (258-dim, 33-channel shipped fit) ────────────────────────────────
N_PHI_FULLCHAN = 258
TRIG_PAIRS = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
DICT_SPEC_FULLCHAN = ('x_full 33-ch; phi = [1, delta(3) linear-observable-only, '
                      'd21=d2-d1, d31=d3-d1, omega(3), Pm(3), '
                      'cos/sin(delta_i-delta_j) for all 6 ordered machine '
                      'pairs (12 cols), Id(3), Iq(3) [exact static dq '
                      'currents from the network solve], Vm(9), Th(9) '
                      '[plain observables, raw delta/theta in NO monomial], '
                      'degree-2 monomials of v20 = {d21, d31, omega, Pm, '
                      'the 12 trig columns}]')


@torch.no_grad()
def algebraics(S, X):
    """X: (N,9) differential [delta(3), omega(3), Pm(3)] numpy float64 ->
    (Id (N,3), Iq (N,3), Vm (N,9), Th (N,9)) numpy. EXACT static map from the
    generator's own pre-fault network solve (kron-reduced Vrec/Yred), no
    integration."""
    dlt = torch.tensor(X[:, 0:3], dtype=torch.float64).to(S.cdtype)
    E = S.Ec.reshape(1, 3) * torch.exp(1j * dlt)                # (N,3)
    Yr = S.Yred_pre[0]          # (3,3) kron-reduced pre-fault (batched)
    Vr = S.Vrec_pre[0]          # (9,3): V_bus = Vr @ E'
    Ig = torch.einsum('ij,bj->bi', Yr, E)                       # (N,3)
    Vb = torch.einsum('ij,bj->bi', Vr, E)                       # (N,9)
    rot = torch.exp(1j * (np.pi / 2 - dlt))    # exp(-1j*(dl-pi/2))
    return (Ig * rot).real.numpy(), (Ig * rot).imag.numpy(), \
        Vb.abs().numpy(), Vb.angle().numpy()


def phi_fullchan(S, X):
    """X: (N,9) = [delta(3), omega(3), Pm(3)] -> (N,258)."""
    dlt, w, pm = X[:, 0:3], X[:, 3:6], X[:, 6:9]
    d21 = (dlt[:, 1] - dlt[:, 0])[:, None]
    d31 = (dlt[:, 2] - dlt[:, 0])[:, None]
    trig = _pair_trig(dlt, TRIG_PAIRS)                          # (N,12)
    v20 = np.concatenate([d21, d31, w, pm, trig], axis=1)       # (N,20)
    mon = np.stack([np.prod(v20[:, c], axis=1)
                    for c in mono_indices(20, 2)], axis=1)      # (N,210)
    Id, Iq, Vm, Th = algebraics(S, np.concatenate([dlt, w, pm], axis=1))
    return np.concatenate([np.ones((X.shape[0], 1)), dlt, v20,
                           Id, Iq, Vm, Th, mon], axis=1)


# ── bare (58-dim Korda-native bare-swing refit) ───────────────────────────────
N_PHI_BARE = 58


def phi_bare(X):
    """X: (N,6) = [delta(3), omega(3)] -> (N,58). Angdiff recipe minus the Pm
    columns: v9 = [d21, d31, omega(3), cos d21, sin d21, cos d31, sin d31]."""
    dlt, w = X[:, 0:3], X[:, 3:6]
    d21 = (dlt[:, 1] - dlt[:, 0])[:, None]
    d31 = (dlt[:, 2] - dlt[:, 0])[:, None]
    v9 = np.concatenate([d21, d31, w,
                         np.cos(d21), np.sin(d21),
                         np.cos(d31), np.sin(d31)], axis=1)
    mon = np.stack([np.prod(v9[:, c], axis=1)
                    for c in mono_indices(9, 2)], axis=1)
    return np.concatenate([np.ones((X.shape[0], 1)), dlt, v9, mon], axis=1)


def x_from_z_phys(Z):
    """Recover the 9 differential states from the angdiff/fullchan lifts."""
    return np.concatenate([Z[:, 1:4], Z[:, 6:9], Z[:, 9:12]], axis=1)


def x_from_z_bare(Z):
    """Recover the 6 bare-swing states from the bare lift."""
    return np.concatenate([Z[:, 1:4], Z[:, 6:9]], axis=1)


# ── bare-swing dynamics (prediction refit generator) ──────────────────────────

def rhs_bare(x6, M, N, Pset, S):
    dlt, w = x6[:, 0:3], x6[:, 3:6]
    th = dlt + S.phase
    dth = th.unsqueeze(2) - th.unsqueeze(1)
    Pe = (M * torch.cos(dth) + N * torch.sin(dth)).sum(-1)
    dw = w - 1.0
    return torch.cat([S.wb * dw,
                      (Pset - Pe - S.D * dw) / (2.0 * S.H)], dim=1)


def rk4_bare(x6, M, N, Pset, h, S):
    k1 = rhs_bare(x6, M, N, Pset, S)
    k2 = rhs_bare(x6 + 0.5 * h * k1, M, N, Pset, S)
    k3 = rhs_bare(x6 + 0.5 * h * k2, M, N, Pset, S)
    k4 = rhs_bare(x6 + h * k3, M, N, Pset, S)
    return x6 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def bank_states(S, seed, n_ep, L, dt):
    """Pre-fault episodes with random constant u -> natural state distribution.
    Returns states (n_ep, L+1, 9) float64 numpy and the episode u's (n_ep, 3).
    (Task-1-validated physics: IEEE9 exact torch swing+governor, RK4 @ 5 ms.)"""
    rng = np.random.default_rng(seed)
    Ypre = S.Yred_pre.expand(n_ep, 3, 3)
    Mp, Np = S.pe_consts(Ypre)
    # per-episode u: 20 % zero, 80 % drawn 50/50 small vs +-1.2 rail
    n_zero = n_ep // 5
    n_rest = n_ep - n_zero
    n_h = n_rest // 2
    u_ep = np.concatenate([
        rng.uniform(-0.05, 0.05, (n_h, 3)),
        rng.uniform(-1.2, 1.2, (n_rest - n_h, 3)),
        np.zeros((n_zero, 3))])
    rng.shuffle(u_ep)
    # initial states: 50 % tight / 50 % wide mixture around equilibrium
    h = n_ep // 2
    w = n_ep - h
    dd = np.concatenate([rng.uniform(-0.10, 0.10, (h, 3)),
                         rng.uniform(-1.0, 1.0, (w, 3))])
    dw = np.concatenate([rng.uniform(-5e-3, 5e-3, (h, 3)),
                         rng.uniform(-0.05, 0.05, (w, 3))])
    dpm = np.concatenate([rng.uniform(-0.3, 0.3, (h, 3)),
                          rng.uniform(-0.6, 0.6, (w, 3))])
    for a in (dd, dw, dpm):
        rng.shuffle(a)
    x0 = np.concatenate([S.delta0.numpy() + dd, 1.0 + dw,
                         S.Pm0.numpy() + dpm], axis=1)
    states = np.empty((n_ep, L + 1, 9))
    states[:, 0] = x0
    xt = torch.tensor(x0, dtype=torch.float64)
    Ps = S.Pm0.unsqueeze(0).expand(n_ep, 3) \
        + torch.tensor(u_ep, dtype=torch.float64)
    for i in range(L):
        xt = S.rk4_real(xt, Mp, Np, Ps, dt)
        states[:, i + 1] = xt.numpy()
    return states, u_ep


@torch.no_grad()
def bank_states_bare(S, seed, n_ep, L, dt):
    """Bare-swing analogue of bank_states: same seed/draw structure (per-episode
    constant u mixture and shuffle; x0 tight/wide mixture; shuffled); governor
    lag dropped: Pset = Pm0 + u_ep held, 6-state RK4."""
    rng = np.random.default_rng(seed)
    Ypre = S.Yred_pre.expand(n_ep, 3, 3)
    Mp, Np = S.pe_consts(Ypre)
    n_zero = n_ep // 5
    n_rest = n_ep - n_zero
    n_h = n_rest // 2
    u_ep = np.concatenate([
        rng.uniform(-0.05, 0.05, (n_h, 3)),
        rng.uniform(-1.2, 1.2, (n_rest - n_h, 3)),
        np.zeros((n_zero, 3))])
    rng.shuffle(u_ep)
    h = n_ep // 2
    w = n_ep - h
    dd = np.concatenate([rng.uniform(-0.10, 0.10, (h, 3)),
                         rng.uniform(-1.0, 1.0, (w, 3))])
    dw = np.concatenate([rng.uniform(-5e-3, 5e-3, (h, 3)),
                         rng.uniform(-0.05, 0.05, (w, 3))])
    for a in (dd, dw):
        rng.shuffle(a)
    x0 = np.concatenate([S.delta0.numpy() + dd, 1.0 + dw], axis=1)
    states = np.empty((n_ep, L + 1, 6))
    states[:, 0] = x0
    xt = torch.tensor(x0, dtype=torch.float64)
    Ps = S.Pm0.unsqueeze(0).expand(n_ep, 3) \
        + torch.tensor(u_ep, dtype=torch.float64)
    for i in range(L):
        xt = rk4_bare(xt, Mp, Np, Ps, dt, S)
        states[:, i + 1] = xt.numpy()
    return states, u_ep


def fresh_u_tuples(rng, states, eps, n, ep_len):
    """Fresh-u tuple sampler: rng draw order (episode idx, time, small-u block,
    rail-u block, shuffle) identical to the validated scripts."""
    e = eps[rng.integers(0, len(eps), size=n)]
    t = rng.integers(0, ep_len, size=n)
    X = states[e, t].astype(np.float64)
    n_small = n // 2
    u = np.concatenate([rng.uniform(-0.02, 0.02, (n_small, 3)),
                        rng.uniform(-1.2, 1.2, (n - n_small, 3))])
    rng.shuffle(u, axis=0)
    return X, u


@torch.no_grad()
def algebraic_crosscheck(S, d):
    """Compare the static algebraic map (computed from the DATASET's own noisy
    differential channels 1..9) against the dataset's stored algebraic channels
    10..33 on PRE-FAULT rows of subsampled validation scenarios."""
    noise = {'Id': 0.005, 'Iq': 0.005, 'Vm': 0.001, 'Th': 0.01}
    tol = {'Id': 0.05, 'Iq': 0.05, 'Vm': 0.005, 'Th': 0.03}
    split2 = np.where(d['split'] == 2)[0]
    errs = {'Id': [], 'Iq': [], 'Vm': [], 'Th': []}
    used = []
    for k in split2[::25]:
        st = np.asarray(d['X'][k], dtype=np.float64)
        m = d['t_grid'] < d['t_fault'][k] - 1e-9
        if m.sum() < 5:
            continue
        used.append(int(k))
        x9 = np.concatenate([st[m][:, D_IDX], st[m][:, W_IDX],
                             st[m][:, PM_IDX]], axis=1)
        Id, Iq, Vm, Th = algebraics(S, x9)
        errs['Id'].append(Id - st[m][:, 9:15][:, [0, 2, 4]])
        errs['Iq'].append(Iq - st[m][:, 9:15][:, [1, 3, 5]])
        errs['Vm'].append(Vm - st[m][:, VM_IDX])
        errs['Th'].append(np.angle(np.exp(1j * (Th - st[m][:, TH_IDX]))))
    res = {kk: float(np.sqrt(np.mean(np.concatenate(vv) ** 2)))
           for kk, vv in errs.items()}
    res['scenarios_checked'] = used
    res['noise_floors'] = noise
    res['tolerances'] = tol
    res['map_check_pass'] = bool(
        all(res[kk] <= tol[kk] for kk in ('Id', 'Iq', 'Vm', 'Th')))
    return res


def _fit_rollout_configs(S, rng, states, roll_args, val_eps, phi, x_from_z,
                         integrator, state_dim, A, B, hz_list):
    """200-step free rollouts, held-out states; SAME rng draw order as the
    shipped fits (r_eps, r_t, Ps_rand(300,1,3), Ps_rail(300,1,3),
    u_rand(300,200,3)). Returns config dict (with horizon prefixes)."""
    n_r = roll_args['n_roll_states']
    r_eps = rng.choice(val_eps, size=n_r, replace=True)
    r_t = rng.integers(0, roll_args['ep_len'], size=n_r)
    X0 = states[r_eps, r_t].astype(np.float64)
    Z0 = phi(S, X0)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n_r, 3, 3))
    rng.uniform(-0.1, 0.1, (n_r, 1, 3))     # Ps_rand (draw consumed, unused)
    Ps_rail = torch.tensor(rng.uniform(-1.2, 1.2, (n_r, 1, 3)),
                           dtype=torch.float64)
    U_cfgs = {'u_zero': np.zeros((n_r, roll_args['roll_steps'], 3)),
              'u_rand': rng.uniform(-0.1, 0.1,
                                    (n_r, roll_args['roll_steps'], 3)),
              'u_rail': np.repeat(Ps_rail.numpy(), roll_args['roll_steps'],
                                  axis=1)}
    roll_cfgs = {}
    for tag, U in U_cfgs.items():
        true = np.empty((n_r, roll_args['roll_steps'] + 1, state_dim))
        true[:, 0] = X0
        xt = torch.tensor(X0, dtype=torch.float64)
        Pm0e = S.Pm0.unsqueeze(0).expand(n_r, 3)
        for k in range(roll_args['roll_steps']):
            xt = integrator(xt, Mp, Np,
                            Pm0e + torch.tensor(U[:, k],
                                                dtype=torch.float64),
                            roll_args['dt'])
            true[:, k + 1] = xt.numpy()
            xt = xt.detach().clone()
        # model rollout: z_{k+1} = A z_k + B u_k
        z = Z0.copy()
        hat = np.empty_like(true)
        hat[:, 0] = X0
        for k in range(roll_args['roll_steps']):
            z = z @ A.T + U[:, k] @ B.T
            hat[:, k + 1] = x_from_z(z)
        err = hat - true
        cfg = {'omega_rmse_1s': float(np.sqrt(np.mean(err[:, :, 3:6] ** 2))),
               'omega_max_abs_1s': float(np.abs(err[:, :, 3:6]).max())}
        if state_dim == 9:
            cfg['delta_rmse_1s'] = float(np.sqrt(np.mean(err[:, :, 0:3] ** 2)))
            cfg['Pm_rmse_1s'] = float(np.sqrt(np.mean(err[:, :, 6:9] ** 2)))
            cfg['true_max_ddelta_drift_rad'] = float(
                np.abs(true[:, -1, 0:3] - S.delta0.numpy()[None, :]).max())
        # horizon-prefix metrics (pooled window INCLUDES the k=0 zero frame —
        # the task2_horizon_curve / task2_edmd_fullchan convention)
        hz = {}
        for HH in hz_list:  # noqa: F821  (arg defined below)
            e0 = err[:, :HH + 1]
            e1 = e0[:, 1:]
            om_traj = np.sqrt(np.mean(e0[:, :, 3:6] ** 2, axis=(1, 2)))
            hz[str(HH)] = {
                'omega_rms_pooled_window':
                    float(np.sqrt(np.mean(e0[:, :, 3:6] ** 2))),
                'omega_rms_mean': float(om_traj.mean()),
                'omega_rms_worst': float(om_traj.max()),
                'omega_rms_pooled_window_excl_step0':
                    float(np.sqrt(np.mean(e1[:, :, 3:6] ** 2))),
                'omega_max_abs': float(np.abs(e0[:, :, 3:6]).max()),
                'ms': float(HH * roll_args['dt'] * 1000)}
        cfg['horizons'] = hz
        roll_cfgs[tag] = cfg
        print(f"rollout[{tag:7s}] omega RMSE(1 s) = "
              f"{cfg['omega_rmse_1s']:.3e}   max|domega| = "
              f"{cfg['omega_max_abs_1s']:.3e}")
    return roll_cfgs


@torch.no_grad()
def cmd_fit_edmd(args, out_dir):
    """Shared seed-42 EDMD protocol; --fit selects the shipped dictionary.
    angdiff2 -> task2_edmd_angdiff.json · fullchan -> task2_edmd_fullchan.json
    · bare -> task34_bare_refit.json."""
    t0 = time.time()
    fit = args.fit
    d = load_data()
    S = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
              os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
              dtype=torch.float64, lines=d['lines'])
    rng = np.random.default_rng(args.seed)
    cfg = {
        'dt': args.dt, 'n_train': int(args.n_train), 'n_val': int(args.n_val),
        'n_episodes': args.n_episodes, 'episode_len': args.ep_len,
        'roll_steps': args.roll_steps, 'n_roll_states': args.n_roll_states,
        'seed': args.seed, 'fit': fit,
    }

    # bank + split + tuples per variant
    if fit == 'bare':
        states, _ = bank_states_bare(S, args.seed, args.n_episodes,
                                     args.ep_len, args.dt)
        state_dim = 6
        n_phi = N_PHI_BARE
        phi = lambda S_, X: phi_bare(X)             # noqa: E731
        x_from_z = x_from_z_bare
        integrator = lambda xt, Mp, Np, Ps, dt: rk4_bare(  # noqa: E731
            xt, Mp, Np, Ps, dt, S)
        hz_list = (10, 20, 40, args.roll_steps)
    else:
        states, _ = bank_states(S, args.seed, args.n_episodes, args.ep_len,
                                args.dt)
        state_dim = 9
        hz_list = (10, 20, 40, 80, 160)
        if fit == 'angdiff2':
            phi = lambda S_, X: phi_angdiff(X)                   # noqa: E731
            n_phi = N_PHI_ANGDIFF
        else:                       # fullchan (33-channel shipped fit)
            phi = lambda S_, X: phi_fullchan(S, X)               # noqa: E731
            n_phi = N_PHI_FULLCHAN
        x_from_z = x_from_z_phys
        integrator = lambda xt, Mp, Np, Ps, dt: S.rk4_real(  # noqa: E731
            xt, Mp, Np, Ps, dt)
    n_ep = states.shape[0]

    # 90/10 split BY EPISODE (adjacent states are near-duplicates; a per-tuple
    # split would leak) -- same rng stream order as the shipped fits
    n_val_ep = max(1, int(round(0.1 * n_ep)))
    val_eps = np.sort(rng.choice(n_ep, size=n_val_ep, replace=False))
    train_eps = np.setdiff1d(np.arange(n_ep), val_eps)

    def sample_tuples(eps, n):
        X, u = fresh_u_tuples(rng, states, eps, n, args.ep_len)
        Mp, Np = S.pe_consts(S.Yred_pre.expand(n, 3, 3))
        Ps = S.Pm0.unsqueeze(0).expand(n, 3) \
            + torch.tensor(u, dtype=torch.float64)
        if fit == 'bare':
            Xp = rk4_bare(torch.tensor(X, dtype=torch.float64),
                          Mp, Np, Ps, args.dt, S).numpy()
        else:
            Xp = S.rk4_real(torch.tensor(X, dtype=torch.float64),
                            Mp, Np, Ps, args.dt).numpy()
        return X, u, Xp

    Xtr, Utr, Xptr = sample_tuples(train_eps, args.n_train)
    Xva, Uva, Xpva = sample_tuples(val_eps, args.n_val)

    # fullchan pre-fit gate: algebraic map cross-check vs the dataset
    alg_check = None
    if fit == 'fullchan':
        alg_check = algebraic_crosscheck(S, d)
        print('algebraic map vs dataset (pre-fault):',
              json.dumps({k: v for k, v in alg_check.items()
                          if k != 'scenarios_checked'}))
        if not alg_check['map_check_pass']:
            print('ALGEBRAIC MAP CROSS-CHECK FAILED -- aborting before fit')
            raise SystemExit(2)

    # one least-squares: z+ = A z + B u; constant row pinned
    if fit == 'bare':
        Ztr, Zptr = phi_bare(Xtr), phi_bare(Xptr)
    else:
        Ztr, Zptr = phi(S, Xtr), phi(S, Xptr)
    D = np.concatenate([Ztr, Utr], axis=1)
    sv = np.linalg.svd(D, compute_uv=False)
    design_cond = float(sv[0] / sv[-1])
    coef, *_ = np.linalg.lstsq(D, Zptr, rcond=None)
    A = coef[:n_phi, :].T.copy()
    B = coef[n_phi:, :].T.copy()
    A[0, :] = 0.0; A[0, 0] = 1.0; B[0, :] = 0.0
    del Ztr, Zptr, D, sv
    spectral_radius = float(np.abs(np.linalg.eigvals(A)).max())

    # one-step prediction on the held-out tuples
    if fit == 'bare':
        Zva = phi_bare(Xva)
        Zp_hat = Zva @ A.T + Uva @ B.T
        e = x_from_z_bare(Zp_hat) - Xpva
        e_trig = Zp_hat[:, 9:13] - phi_bare(Xpva)[:, 9:13]
        one_step = {'rmse_all': float(np.sqrt(np.mean(e ** 2))),
                    'rmse_omega': float(np.sqrt(np.mean(e[:, 3:6] ** 2))),
                    'rmse_delta': float(np.sqrt(np.mean(e[:, 0:3] ** 2))),
                    'max_abs_omega': float(np.abs(e[:, 3:6]).max()),
                    'rmse_trig_columns': float(np.sqrt(np.mean(e_trig ** 2)))}
    else:
        Zva = phi(S, Xva)
        Zp_hat = Zva @ A.T + Uva @ B.T
        Xp_hat = x_from_z_phys(Zp_hat)
        e = Xp_hat - Xpva
        phi_pva = phi(S, Xpva)
        e_trig = Zp_hat[:, 12:16 if fit == 'angdiff2' else 24] \
            - phi_pva[:, 12:16 if fit == 'angdiff2' else 24]
        one_step = {'rmse_all': float(np.sqrt(np.mean(e ** 2))),
                    'rmse_omega': float(np.sqrt(np.mean(e[:, 3:6] ** 2))),
                    'rmse_delta': float(np.sqrt(np.mean(e[:, 0:3] ** 2))),
                    'rmse_Pm': float(np.sqrt(np.mean(e[:, 6:9] ** 2))),
                    'max_abs_omega': float(np.abs(e[:, 3:6]).max()),
                    'rmse_trig_columns': float(np.sqrt(np.mean(e_trig ** 2))),
                    'true_increment_omega_rms': float(np.sqrt(np.mean(
                        (Xpva[:, 3:6] - Xva[:, 3:6]) ** 2)))}
        if fit == 'fullchan':
            one_step['rmse_alg_blocks'] = float(np.sqrt(np.mean(
                (Zp_hat[:, 24:48] - phi_pva[:, 24:48]) ** 2)))
    print('one-step:', json.dumps(one_step))
    del Zva, Zp_hat

    # 200-step free rollouts (first 1 s), held-out states
    roll_args = {'n_roll_states': cfg['n_roll_states'],
                 'ep_len': cfg['episode_len'],
                 'roll_steps': cfg['roll_steps'], 'dt': cfg['dt']}
    roll_cfgs = _fit_rollout_configs(S, rng, states, roll_args, val_eps, phi,
                                     x_from_z, integrator, state_dim, A, B,
                                     hz_list)

    worst = float(max(c['omega_rmse_1s'] for c in roll_cfgs.values()))
    gate = {'free_roll_omega_rmse_worst': worst,
            'deadband': 0.005,
            'gate_pass_1e-3': bool(worst <= 1e-3),
            'worst_config': max(roll_cfgs, key=lambda k:
                                roll_cfgs[k]['omega_rmse_1s'])}
    print('gate:', json.dumps(gate))

    note_common = ('EDMD fitted on PRE-FAULT+governor dynamics, '
                   'hypothesis-blind, exactly Korda design (Korda/Susuki/'
                   'Mezic 2018, arXiv:1803.10744).')
    artifacts = {
        'cmd': ' '.join(sys.argv), 'note': note_common,
        'n_phi': int(n_phi), 'seed': args.seed,
        'seeds_used': {'np_random_generator': args.seed},
        'config': {
            **{k: cfg[k] for k in ('dt', 'n_train', 'n_val', 'n_episodes',
                                   'episode_len')},
            'n_phi': int(n_phi),
            'dictionary': {
                'angdiff2': DICT_SPEC_ANGDIFF,
                'fullchan': DICT_SPEC_FULLCHAN,
                'bare': 'angdiff recipe minus Pm columns: v9 = [d21, d31, '
                        'omega(3), cos d21, sin d21, cos d31, sin d31]; '
                        'phi = [1, delta(3) linear-observable-only, v9, '
                        'deg-2 monomials of v9] -> n_phi = 58; u = dPm '
                        'offset on the mechanical-power channel '
                        '(Korda-native fraction-of-nominal-power semantics)',
            }[fit],
            'state_x': ('(delta, omega) per machine = [d0 w0 d1 w1 d2 w2]'
                        if fit == 'bare' else
                        '(delta, omega, Pm) per machine '
                        '= [d0 w0 d1 w1 d2 w2 pm0 pm1 pm2]'),
            'u_range': [-1.2, 1.2],
            'u_mixture': '50% U(-0.02,0.02) + 50% U(-1.2,1.2) per column',
            'split': '90/10 by episode, seeded',
            'roll_steps': args.roll_steps, 'roll_states': args.n_roll_states,
            'consolidated_from': {
                'angdiff2': 'bus_9/_koopa_edmd_angdiff.py',
                'fullchan': 'bus_9/_koopa_edmd_fullchan.py',
                'bare': 'bus_9/_koopa_mpc.py:run_bare'}[fit],
        },
        'design_condition_number': design_cond,
        'spectral_radius': spectral_radius,
        'A': A.tolist(), 'B': B.tolist(),
        'one_step': one_step, 'rollouts': roll_cfgs, 'gate': gate,
        'runtime_s': time.time() - t0,
    }
    if fit == 'bare':
        artifacts['config']['generator'] = \
            'bare-swing RK4 (rhs_bare/rk4_bare): exact torch physics, ' \
            'governor lag removed, Pset = Pm0 + u per tuple'
        artifacts.pop('design_condition_number')
    if alg_check is not None:
        artifacts['algebraic_map_crosscheck_vs_dataset'] = alg_check
    fname = {'angdiff2': 'task2_edmd_angdiff.json',
             'fullchan': 'task2_edmd_fullchan.json',
             'bare': 'task34_bare_refit.json'}[fit]
    path = os.path.join(out_dir, fname)
    if fit == 'bare':            # shipped self-hash convention
        blob = json.dumps(artifacts, indent=1)
        artifacts['sha256_self'] = hashlib.sha256(blob.encode()).hexdigest()
    with open(path, 'w') as f:
        json.dump(artifacts, f, indent=1)
    print(f"saved {path}  ({time.time() - t0:.1f} s)")


# ══════════════════════════════════════════════════════════════════════════════
# Lifts (fit JSON -> predictor used by KF / QP MPC)
# ══════════════════════════════════════════════════════════════════════════════

class GovLift:
    """Fullchan (258-dim) lift; u = dP offset on the governed columns."""
    tag = 'fullchan'
    src_json = 'task2_edmd_fullchan.json'
    n_state = 9

    def __init__(self, S, out_dir):
        self.S = S
        p = find_artifact(self.src_json, out_dir)
        j = json.load(open(p))
        self.A = np.array(j['A']); self.B = np.array(j['B'])
        self.src_sha = sha256_of(p)

    def phi(self, X):
        """X: (N, 9) -> (N, 258)"""
        return phi_fullchan(self.S, X)

    def phys_from_z(self, z):
        return np.concatenate([z[1:4], z[6:9], z[9:12]])

    def phys_rows(self, Z):
        return np.concatenate([Z[:, 1:4], Z[:, 6:9], Z[:, 9:12]], axis=1)

    def phys_state(self, x9):
        return x9

    def omega_rows(self):
        return slice(6, 9)


class BareLift:
    """Authorized bare-swing refit (58-dim); u = dPm offset (mechanical-power
    channel); states (delta, omega) 6-dim."""
    tag = 'bare'
    src_json = 'task34_bare_refit.json'
    n_state = 6

    def __init__(self, out_dir):
        p = find_artifact(self.src_json, out_dir)
        j = json.load(open(p))
        self.A = np.array(j['A']); self.B = np.array(j['B'])
        self.src_sha = j.get('sha256_self')

    def phi(self, X):
        """X: (N, 6) = [delta(3), omega(3)] -> (N, 58)."""
        return phi_bare(X)

    def phys_from_z(self, z):
        return np.concatenate([z[1:4], z[6:9]])

    def phys_rows(self, Z):
        return np.concatenate([Z[:, 1:4], Z[:, 6:9]], axis=1)

    def phys_state(self, x9):
        return x9[:6]

    def omega_rows(self):
        return slice(6, 9)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario / plant (task-1-validated generator conventions)
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    d = load_data()
    S = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
              os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
              dtype=torch.float64, lines=d['lines'])
    return d, S


def fault_arrays(S, d, k):
    h = scenario_hyp(S, d, k)
    a = torch.tensor([d['meta_alpha'][k]], dtype=torch.float64)
    Yf, _ = S.kron_reduce(S.faulted_Y(h, a))
    dPm = S.dPm0(h, a)[0]
    return Yf, dPm


def run_true_plant(S, d, k, uplane=None):
    """True closed-loop plant: exact RK4 @ 5 ms, midpoint fault mask,
    Pset = Pm0 + dPm0*act + u_applied(t). uplane: (T-1, 3) numpy or None.
    Returns clean states (T, 9) numpy; convention identical to
    simulate_clean (verified there vs the stored dataset)."""
    T = len(d['t_grid'])
    Yf, dPm = fault_arrays(S, d, k)
    Mp, Np = S.pe_consts(S.Yred_pre)
    Mf, Nf = S.pe_consts(Yf)
    tf, tc = d['t_fault'][k], d['t_clear'][k]
    x = torch.cat([S.delta0, torch.ones(3), S.Pm0])
    states = np.empty((T, 9))
    states[0] = x.numpy()
    Pm0e = S.Pm0
    for t in range(T - 1):
        tm = t * DT + DT / 2
        act = 1.0 if (tm >= tf - 1e-9 and tm < tc - 1e-9) else 0.0
        M, N = (Mf, Nf) if act else (Mp, Np)
        Pset = Pm0e + dPm * act
        if uplane is not None and np.any(uplane[t] != 0.0):
            Pset = Pset + torch.tensor(uplane[t], dtype=torch.float64)
        x = S.rk4_real(x.view(1, 9), M, N, Pset, DT)[0]
        states[t + 1] = x.numpy()
    return states


def measure(xnp, m, rng):
    """Noisy measurement of state xnp (9,) as m-dim vector:
    m=9 -> [delta(3), omega(3), Pm(3)]; m=6 -> [delta(3), omega(3)]."""
    y = np.concatenate([xnp[0:3] + rng.normal(0, .01, 3),
                        xnp[3:6] + rng.normal(0, 5e-5, 3)])
    if m == 9:
        y = np.concatenate([y, xnp[6:9] + rng.normal(0, 5e-3, 3)])
    return y


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — steady lifted Kalman filter (physical-state form, DARE gain)
# ══════════════════════════════════════════════════════════════════════════════

def phys_jacobian(gstep, x0eq):
    F = np.empty((len(x0eq), len(x0eq)))
    f0 = gstep(x0eq)
    hj = 1e-6
    for c in range(len(x0eq)):
        xp = x0eq.copy(); xp[c] += hj
        F[:, c] = (gstep(xp) - f0) / hj
    return F


def steady_kf(F, C, R, q):
    """One DARE -> steady gain; returns K and the closed-loop error spectrum."""
    P = sla.solve_discrete_are(F.T, C.T, q * np.eye(F.shape[0]), R)
    K = F @ P @ C.T @ np.linalg.solve(C @ P @ C.T + R, np.eye(C.shape[0]))
    acl = (np.eye(F.shape[0]) - K @ C) @ F
    ev = np.sort(np.abs(np.linalg.eigvals(acl)))[::-1]
    return dict(K=K, acl_top_eigs=[float(v) for v in ev[:6]])


def make_kf(lift, m, S):
    """Steady physical-state KF for one variant: DARE on the equilibrium
    linearization of the re-lifted predictor phys(A phi(x) + B u)."""
    C = np.eye(m)
    R = np.diag(NOISE_STD[:m] ** 2)

    def gstep(x):
        u = np.zeros(3)
        return lift.phys_from_z(lift.A @ lift.phi(x[None, :])[0]
                                + lift.B @ u)

    x0eq = np.concatenate([S.delta0.numpy(), np.ones(3),
                           S.Pm0.numpy()])[:m]
    F = phys_jacobian(gstep, x0eq)
    info = steady_kf(F, C, R, KF_Q)
    info['F'] = F
    return info


@torch.no_grad()
def cmd_fit_kf(args, out_dir):
    t0 = time.time()
    d, S = load_all()
    lift = GovLift(S, out_dir)
    A, B = lift.A, lift.B
    kfinfo = make_kf(lift, 9, S)
    K = kfinfo['K']
    print('KF design: q =', KF_Q, 'K diag:', np.round(np.diag(K), 4),
          'rho(A_cl) top-3:', np.round(kfinfo['acl_top_eigs'][:3], 6))

    # pre-fault sizing sanity: blind-observer (u=0 prediction) tracking on
    # fresh pre-fault u-rail episodes (u unknown to the observer)
    rng = np.random.default_rng(101)
    ga_n, ga_L = 200, 200
    u_ep = rng.uniform(-1.2, 1.2, (ga_n, 3))
    dd = rng.uniform(-0.10, 0.10, (ga_n, 3)); rng.shuffle(dd)
    dw = rng.uniform(-5e-3, 5e-3, (ga_n, 3)); rng.shuffle(dw)
    x0 = np.concatenate([S.delta0.numpy() + dd, 1.0 + dw,
                         np.tile(S.Pm0.numpy(), (ga_n, 1))], axis=1)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(ga_n, 3, 3))
    Pm0e = S.Pm0.unsqueeze(0).expand(ga_n, 3)
    xt = torch.tensor(x0, dtype=torch.float64)
    true_b = np.empty((ga_n, ga_L + 1, 9)); true_b[:, 0] = x0
    for t in range(ga_L):
        xt = S.rk4_real(xt, Mp, Np,
                        Pm0e + torch.tensor(u_ep, dtype=torch.float64), DT)
        true_b[:, t + 1] = xt.numpy()
    rngy = np.random.default_rng(202)
    track = []
    for ep in range(40):
        Tb = true_b[ep]
        Yb = np.concatenate([Tb[:, 0:3] + rngy.normal(0, .01, (ga_L + 1, 3)),
                             Tb[:, 3:6] + rngy.normal(0, 5e-5, (ga_L + 1, 3)),
                             Tb[:, 6:9] + rngy.normal(
                                 0, 5e-3, (ga_L + 1, 3))], axis=1)
        xh = Tb[0].copy()
        earr = np.empty((ga_L + 1, 9)); earr[0] = 0.0
        for t in range(ga_L):
            xp = lift.phys_from_z(A @ lift.phi(xh[None, :])[0])
            xh = xp + K @ (Yb[t + 1] - xp)
            earr[t + 1] = xh - Tb[t + 1]
        track.append(float(np.sqrt(np.mean(earr[:, 3:6] ** 2))))
    sizing = {'prefault_blind_rail_omega_rmse_mean': float(np.mean(track)),
              'prefault_blind_rail_omega_rmse_worst': float(np.max(track)),
              'note': 'u unknown to the observer (rails +-1.2 pu), 1 s '
                      'windows; this is the NON-LEAKY Q-sized sanity probe '
                      '(fault-free).'}
    print('pre-fault blind-rail tracking omega RMSE mean/worst:',
          f'{sizing["prefault_blind_rail_omega_rmse_mean"]:.2e} /',
          f'{sizing["prefault_blind_rail_omega_rmse_worst"]:.2e}')

    # faulted scenarios (held-out: split==2, never in the EDMD bank)
    scen_out = []
    for k in SCEN_ORDER:
        sim = run_true_plant(S, d, k)            # u = 0 blind plant
        rng = np.random.default_rng(42)
        T = sim.shape[0]
        Y = np.empty((T, 9))
        Y[:, 0:3] = sim[:, 0:3] + rng.normal(0, .01, (T, 3))
        Y[:, 3:6] = sim[:, 3:6] + rng.normal(0, 5e-5, (T, 3))
        Y[:, 6:9] = sim[:, 6:9] + rng.normal(0, 5e-3, (T, 3))
        xhat = np.empty((T, 9)); xhat[0] = sim[0]
        for t in range(T - 1):
            xp = lift.phys_from_z(A @ lift.phi(xhat[t:t + 1])[0])
            xhat[t + 1] = xp + K @ (Y[t + 1] - xp)
        err = xhat - sim
        tf, tc = d['t_fault'][k], d['t_clear'][k]
        tg = d['t_grid']
        msk_f = (tg >= tf - 1e-9) * (tg < tc - 1e-9)
        msk_pre = tg < tf - 1e-9
        msk_post = tg >= tc - 1e-9
        rmse = lambda msk: float(np.sqrt(np.mean(err[msk] ** 2)))  # noqa: E731
        bias = lambda msk: np.mean(err[msk], axis=0).tolist()      # noqa: E731
        rec = {
            'scenario': int(k),
            'type': TYPE_NAMES[d['y_type'][k] - 1],
            'alpha': float(d['meta_alpha'][k]),
            't_fault': float(tf), 't_clear': float(tc),
            'windows': {
                'pre': {'rmse': rmse(msk_pre), 'bias_mean': bias(msk_pre)},
                'fault': {'rmse': rmse(msk_f), 'bias_mean': bias(msk_f)},
                'post': {'rmse': rmse(msk_post), 'bias_mean': bias(msk_post)},
            },
            'per_channel_fault': {
                ch: {'rmse': float(np.sqrt(np.mean(err[msk_f][:, sl] ** 2))),
                     'bias_mean': float(np.mean(err[msk_f][:, sl])),
                     'max_abs': float(np.abs(err[msk_f][:, sl]).max())}
                for ch, sl in [('delta', slice(0, 3)),
                               ('omega', slice(3, 6)),
                               ('Pm', slice(6, 9))]},
            'est_peak_abs_omega_minus_1': float(np.abs(xhat[:, 3:6] - 1).max()),
            'true_peak_abs_omega_minus_1': float(np.abs(sim[:, 3:6] - 1).max()),
            'omega_err_abs_at_fault_steps':
                np.abs(err[msk_f][:, 3:6]).round(8).tolist(),
        }
        scen_out.append(rec)
        print(f"k={k} ({rec['type']}): omega err RMSE "
              f"pre {rec['windows']['pre']['rmse']:.2e} / "
              f"fault {rec['windows']['fault']['rmse']:.2e} / "
              f"post {rec['windows']['post']['rmse']:.2e}; "
              f"est peak {rec['est_peak_abs_omega_minus_1']:.2e} "
              f"vs true {rec['true_peak_abs_omega_minus_1']:.2e}")

    out = {
        'cmd': ' '.join(sys.argv),
        'note': 'Task 3: steady-state observer on measured noisy channels; '
                'fault-time bias RECORDED as a result, not tuned away. '
                'DESIGN (ill-posedness remedy, probe numbers in the shipped '
                'task34_kf.json record): the literal full-lifted steady KF '
                'random-walks the estimate because the lifted A near-unit '
                'artifact modes (rho 1.0067) are only ~1e-3 consistent under '
                'memory iteration (the Task-2 wall). Shipped: physical-state '
                'steady KF -- prediction phys(A phi(xhat) + B u) with the '
                'SAME A the MPC uses, re-lifted after each 5 ms step; '
                'correction = steady gain from ONE DARE on the equilibrium '
                'linearization F = d/dx phys(A phi(x)); C = measured noisy '
                'channels (delta, omega, Pm; SIM layout contiguous, not the '
                '33-channel dataset layout); R = dataset noise stds; Q = '
                '1e-4 I sized NON-LEAKILY from pre-fault blind-observer '
                'u-rail tracking.',
        'fit_tag': lift.tag, 'fit_src_json': lift.src_json,
        'fit_src_sha256': lift.src_sha,
        'kf_design': {'q': KF_Q,
                      'form': 'physical-state, re-lifted predictor',
                      'C_rows': 'delta, omega, Pm (dataset noise stds)',
                      'K_diag': np.diag(K).round(6).tolist(),
                      'acl_top_eigs': kfinfo['acl_top_eigs']},
        'prefault_sizing_check': sizing,
        'scenarios': scen_out,
        'gate': {'bias_recorded_as_result': True},
        'runtime_s': time.time() - t0,
    }
    path = os.path.join(out_dir, 'task34_kf.json')
    with open(path, 'w') as f:
        f.write(json.dumps(out, indent=1))
    print(f'saved {path}  ({time.time() - t0:.1f} s)')


# ══════════════════════════════════════════════════════════════════════════════
# Task 4 — dense-form QP MPC (osqp), --channel {gov_fullchan,korda_native_bare}
# ══════════════════════════════════════════════════════════════════════════════

def build_qp_layout(lift, A, B):
    """Constant QP blocks. Variables x = [u (3*N_DEC); s (H*3); st (3)].
    omega_hat_k = Cw A^k z0 + sum_c T[k,c] u_c, cell c covers model steps
    4c..4c+3. Rows: u bounds, s >= 0, rate |u_c - u_{c-1}| <= rate,
    hinge s -/+ T u >= +/- (of) - db, terminal st -/+ T_H u >= +/- of_H."""
    n = A.shape[0]
    Cw_rows = lift.omega_rows()
    Cw = np.zeros((3, n)); Cw[:, Cw_rows] = np.eye(3)
    Apow = [np.eye(n)]
    for i in range(H):
        Apow.append(Apow[-1] @ A)
    Tkk = np.zeros((H, N_DEC, 3, 3))
    for k in range(1, H + 1):
        for c in range(N_DEC):
            j = k - 1 - DEC_EVERY * c
            if j >= 0:
                Tkk[k - 1, c] = Cw @ Apow[j] @ B

    nvar_u = 3 * N_DEC
    nvar_s = H * 3
    NV = nvar_u + nvar_s + 3
    rows = []        # list of (row_index, var_index, coefficient)
    l_bounds = []; u_bounds = []

    def add(row_entries, l, u):
        r = len(l_bounds)
        for i, cf in row_entries:
            rows.append((r, i, cf))
        l_bounds.append(l); u_bounds.append(u)
        return r

    # u bounds rows
    for i in range(nvar_u):
        add([(i, 1.0)], -QP_UMAX, QP_UMAX)
    i0_s = nvar_u
    for i in range(nvar_s + 3):
        add([(i0_s + i, 1.0)], 0.0, np.inf)
    # rate rows (c == 0 row bounds move with the previously applied u cell)
    rate_rows = []
    for mm in range(3):
        for c in range(N_DEC):
            ents = [(c * 3 + mm, 1.0)]
            if c > 0:
                ents.append(((c - 1) * 3 + mm, -1.0))
            r = add(ents, -QP_RATE, QP_RATE)
            rate_rows.append((c, mm, r))
    # hinge rows
    hinge_rows = []
    for k in range(H):
        for mm in range(3):
            ents = [(i0_s + k * 3 + mm, 1.0)]
            for c in range(N_DEC):
                cf = Tkk[k, c][mm]
                if np.any(cf != 0):
                    ents.extend([(c * 3 + qi, -cf[qi]) for qi in range(3)])
            r_plus = add(ents, 0.0, np.inf)     # l set per decision
            ents = [(i0_s + k * 3 + mm, 1.0)]
            for c in range(N_DEC):
                cf = Tkk[k, c][mm]
                if np.any(cf != 0):
                    ents.extend([(c * 3 + qi, cf[qi]) for qi in range(3)])
            r_minus = add(ents, 0.0, np.inf)
            hinge_rows.append((k, mm, r_plus, r_minus))
    # terminal rows (deadband 0)
    term_rows = []
    i0_st = nvar_u + nvar_s
    th = Tkk[H - 1]
    for mm in range(3):
        ents = [(i0_st + mm, 1.0)]
        ents.extend([(c * 3 + qi, -th[c][mm][qi]) for c in range(N_DEC)
                     for qi in range(3) if th[c][mm][qi] != 0])
        r_plus = add(ents, 0.0, np.inf)
        ents = [(i0_st + mm, 1.0)]
        ents.extend([(c * 3 + qi, th[c][mm][qi]) for c in range(N_DEC)
                     for qi in range(3) if th[c][mm][qi] != 0])
        r_minus = add(ents, 0.0, np.inf)
        term_rows.append((r_plus, r_minus))

    NR = len(l_bounds)
    Av = np.zeros((NR, NV))
    for r, i, cf in rows:
        Av[r, i] = cf
    P = np.zeros((NV, NV))
    for i in range(nvar_u):
        P[i, i] = 2 * QP_QU
    for j in range(nvar_s):
        P[i0_s + j, i0_s + j] = 2 * QP_W
    for j in range(3):
        P[i0_st + j, i0_st + j] = 2 * QP_WT
    qvec = np.zeros(NV)
    return dict(A=sparse.csc_matrix(Av), P=sparse.csc_matrix(P), q=qvec,
                l0=np.array(l_bounds), u0=np.array(u_bounds),
                T=Tkk, Cw=Cw, hinge_rows=hinge_rows, rate_rows=rate_rows,
                term_rows=term_rows, NV=NV)


def predict_omega_free(A, Cw, z0):
    """Free prediction of the omega row over k = 1..H, minus 1: (H, 3)."""
    of = np.empty((H, 3))
    z = z0.copy()
    for i in range(H):
        z = A @ z
        of[i] = Cw @ z - 1.0
    return of


def closed_loop(S, d, k, lift, K, qp, osqp_mod, settings):
    """Floor vs controller on scenario k; true-dynamics peaks + QP timings."""
    A, B, Cw = lift.A, lift.B, qp['Cw']
    l0, u0 = qp['l0'], qp['u0']
    hinge_rows = qp['hinge_rows']; rate_rows = qp['rate_rows']
    term_rows = qp['term_rows']
    m = lift.n_state
    tf, tc = d['t_fault'][k], d['t_clear'][k]

    # floor: no dP
    floor = run_true_plant(S, d, k)
    peak_floor = float(np.max(np.abs(floor[:, 3:6] - 1.0)))

    # controller run
    Yf, dPm = fault_arrays(S, d, k)
    Mp, Np = S.pe_consts(S.Yred_pre)
    Mf, Nf = S.pe_consts(Yf)
    T_grid = len(d['t_grid'])
    rng = np.random.default_rng(42)
    x = torch.cat([S.delta0, torch.ones(3), S.Pm0])
    true = np.empty((T_grid, 9)); true[0] = x.numpy()
    xhat = x.numpy().copy()[:m]
    uplane = np.zeros((T_grid, 3))
    u_prev = np.zeros(3)
    warm_ms, cold_ms, u_cells = [], [], []

    sol = osqp_mod.OSQP()
    sol.setup(P=qp['P'], q=qp['q'], A=qp['A'], l=l0, u=u0, **settings)

    for t in range(T_grid - 1):
        tm = t * DT + DT / 2
        act = 1.0 if (tm >= tf - 1e-9 and tm < tc - 1e-9) else 0.0
        M, N = (Mf, Nf) if act else (Mp, Np)
        Pset = S.Pm0 + dPm * act + torch.tensor(uplane[t],
                                                dtype=torch.float64)
        x = S.rk4_real(x.view(1, 9), M, N, Pset, DT)[0]
        true[t + 1] = x.numpy()
        xnp = x.numpy()
        # KF update on the new measurement (u_applied known to the filter)
        y = measure(xnp, m, rng)
        xp = lift.phys_from_z(A @ lift.phi(xhat[None, :])[0]
                              + B @ uplane[t])
        xhat = xp + K @ (y - xp)
        # decision instant
        if (t + 1) % DEC_EVERY == 0 and (t + 1) <= T_grid - 1 - DEC_EVERY:
            z0 = lift.phi(xhat[None, :])[0]
            of1 = predict_omega_free(A, Cw, z0)              # (H, 3), omega-1
            lu = l0.copy(); uu = u0.copy()
            for (kk, mm, r_plus, r_minus) in hinge_rows:
                lu[r_plus] = of1[kk, mm] - QP_DB
                lu[r_minus] = -of1[kk, mm] - QP_DB
            for im, (r_plus, r_minus) in enumerate(term_rows):
                lu[r_plus] = of1[H - 1, im]
                lu[r_minus] = -of1[H - 1, im]
            for (c, mm1, rrow) in rate_rows:
                if c == 0:
                    lu[rrow] = -QP_RATE + u_prev[mm1]
                    uu[rrow] = +QP_RATE + u_prev[mm1]
            # warm solve (bounds only change; P/A constant; warm start held)
            tw = time.perf_counter()
            sol.update(l=lu, u=uu)
            res = sol.solve()
            warm_ms.append((time.perf_counter() - tw) * 1e3)
            # cold solve: fresh setup + solve of the same problem
            tcold = time.perf_counter()
            sol2 = osqp_mod.OSQP()
            sol2.setup(P=qp['P'], q=qp['q'], A=qp['A'], l=lu, u=uu,
                       **settings)
            res2 = sol2.solve()
            cold_ms.append((time.perf_counter() - tcold) * 1e3)
            ucell = np.clip(res.x[:3 * N_DEC].reshape(N_DEC, 3)[0],
                            -QP_UMAX - 1e-9, QP_UMAX + 1e-9)
            u_cells.append(ucell.tolist())
            u_prev = ucell
            uplane[t + 1:t + 1 + DEC_EVERY] = ucell

    peak_ctrl = float(np.max(np.abs(true[:, 3:6] - 1.0)))
    tg = d['t_grid']
    msk = (tg >= tf - 1e-9) * (tg < tc + 1.0)      # fault window + 1 s grace
    peak_floor_fw = float(np.max(np.abs(floor[msk][:, 3:6] - 1.0)))
    peak_ctrl_fw = float(np.max(np.abs(true[msk][:, 3:6] - 1.0)))
    wp = np.concatenate(u_cells) if u_cells else np.zeros(0)
    return {
        'scenario': int(k), 'type': TYPE_NAMES[d['y_type'][k] - 1],
        'alpha': float(d['meta_alpha'][k]),
        'peak_floor': peak_floor, 'peak_ctrl': peak_ctrl,
        'peak_floor_faultwin_1s': peak_floor_fw,
        'peak_ctrl_faultwin_1s': peak_ctrl_fw,
        'peak_reduction': peak_floor - peak_ctrl,
        'peak_reduction_pct': (peak_floor - peak_ctrl) / peak_floor * 100.0,
        'gate_pass': bool(peak_ctrl < peak_floor),
        'u_max_used': float(np.abs(uplane).max()) if len(u_cells) else 0.0,
        'u_l1_mean_over_decisions': float(np.mean(np.abs(wp)))
            if len(wp) else 0.0,
        'n_decisions': len(warm_ms),
        'qp_ms_warm': [round(x, 4) for x in warm_ms],
        'qp_ms_cold': [round(x, 4) for x in cold_ms],
        'qp_ms_warm_mean': float(np.mean(warm_ms)) if warm_ms else 0.0,
        'qp_ms_warm_median': float(np.median(warm_ms)) if warm_ms else 0.0,
        'qp_ms_warm_max': float(np.max(warm_ms)) if warm_ms else 0.0,
        'qp_ms_cold_mean': float(np.mean(cold_ms)) if cold_ms else 0.0,
        'qp_ms_cold_median': float(np.median(cold_ms)) if cold_ms else 0.0,
        'qp_ms_cold_max': float(np.max(cold_ms)) if cold_ms else 0.0,
        'qp_status_warm_last': str(getattr(res, 'info', res).status)
                              if warm_ms else 'never-solved',
        'u_cells_first20': u_cells[:20],
    }


@torch.no_grad()
def cmd_run_mpc(args, out_dir):
    t0 = time.time()
    import osqp as osqp_mod
    d, S = load_all()
    settings = dict(verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                    max_iter=20000, polishing=True)

    want = ['gov_fullchan', 'korda_native_bare'] \
        if args.channel == 'both' else [args.channel]
    variants = {}
    results = {'variants': {},
               'qp_knobs': dict(w=QP_W, db=QP_DB, Wt=QP_WT, qu=QP_QU,
                                umax=QP_UMAX, rate=QP_RATE, H_steps=H,
                                dec_ms=float(DEC_EVERY * DT * 1000),
                                note='u held within each 20 ms decision cell'),
               'fit_provenance': {}}
    if 'gov_fullchan' in want:
        lift_gov = GovLift(S, out_dir)
        variants['gov_fullchan'] = dict(lift=lift_gov, m=9)
    if 'korda_native_bare' in want:
        variants['korda_native_bare'] = dict(lift=BareLift(out_dir), m=6)

    for vtag, v in variants.items():
        v['kf'] = make_kf(v['lift'], v['m'], S)
        v['qp'] = build_qp_layout(v['lift'], v['lift'].A, v['lift'].B)
        results['fit_provenance'][vtag] = {
            'fit_json': v['lift'].src_json, 'sha256': v['lift'].src_sha,
            'n_phi': int(v['lift'].A.shape[0]),
            'K_diag': np.round(np.diag(v['kf']['K']), 6).tolist()}

    for vtag, v in variants.items():
        vres = []
        for k in SCEN_ORDER:
            rec = closed_loop(S, d, k, v['lift'], v['kf']['K'], v['qp'],
                              osqp_mod, settings)
            vres.append(rec)
            print(f"[{vtag}] k={rec['scenario']} ({rec['type']}): peak floor "
                  f"{rec['peak_floor']:.5f} -> {rec['peak_ctrl']:.5f} "
                  f"({rec['peak_reduction_pct']:+.2f} %)  QP warm "
                  f"{rec['qp_ms_warm_median']:.2f} / cold "
                  f"{rec['qp_ms_cold_median']:.2f} ms  "
                  f"umax {rec['u_max_used']:.3f}  "
                  f"GATE {'PASS' if rec['gate_pass'] else 'FAIL'}")
        results['variants'][vtag] = vres

    gate_rows = []
    for vtag in results['variants']:
        for r in results['variants'][vtag][:2]:
            gate_rows.append(r['gate_pass'])
    results['gate'] = {'gate_pair': SCEN_ORDER[:2],
                       'gate_pair_pass': bool(all(gate_rows))}
    results['cmd'] = ' '.join(sys.argv)
    results['runtime_s'] = time.time() - t0
    path = os.path.join(out_dir, 'task34_mpc.json')
    with open(path, 'w') as f:
        f.write(json.dumps(results, indent=1))
    print('GATE:', json.dumps(results['gate']))
    print(f'saved {path}  ({time.time() - t0:.1f} s)')


# ══════════════════════════════════════════════════════════════════════════════
# Sweep scorer core (true-dynamics, f64 CPU, batched) — shared by
# crosscheck / sweep / scan. Structural replica of verify_true internals.
# ══════════════════════════════════════════════════════════════════════════════

def fault_arrays_row(S, d, ks):
    """Per-row faulted network + dPm0 from METADATA (independent construction,
    same recipe as verify_true.true_network)."""
    n = len(ks)
    Yf = torch.zeros(n, 3, 3, dtype=S.cdtype, device=S.device)
    dPm = torch.zeros(n, 3, dtype=S.rdtype, device=S.device)
    for j, k in enumerate(ks):
        h = HYPOTHESES.index((TYPE_NAMES[int(d['y_type'][k]) - 1],
                              int(d['meta_loc'][k])))
        a = torch.tensor([float(d['meta_alpha'][k])], dtype=S.rdtype,
                         device=S.device)
        Yf[j], _ = S.kron_reduce(S.faulted_Y(h, a))
        dPm[j] = S.dPm0(h, a)[0]
    return Yf, dPm


def full_sim_np(S, tf, tc, tk, t_off, Yf, dPm, dP, dt, n_steps, dP_seq=None,
                want_states=True):
    """True-dynamics full-scenario run, STRUCTURAL REPLICA of
    verify_true._full_sim_ref (op-for-op), extended with
      * dP_seq: (n, n_steps, 3) per-step ΔP array (per-decision sequences,
        already zero outside the applied interval on the plant side), and
      * a float64/device-generic S.
    Returns om (n, n_steps+1, 3) and optionally states (n, n_steps+1, 9)."""
    n = tf.shape[0]
    dev = S.device
    Ypre = S.Yred_pre.expand(n, 3, 3)
    Mp, Np = S.pe_consts(Ypre)
    Mf, Nf = S.pe_consts(Yf)
    x = torch.cat([S.delta0, torch.ones(3, device=dev), S.Pm0]) \
        .unsqueeze(0).expand(n, 9).clone()
    dP_t = dP if dP is not None else torch.zeros(n, 3, dtype=S.rdtype,
                                                 device=dev)
    om = torch.zeros(n, n_steps + 1, 3, dtype=S.rdtype, device=dev)
    om[:, 0] = x[:, 3:6]
    states = torch.zeros(n, n_steps + 1, 9, dtype=S.rdtype, device=dev) \
        if want_states else None
    if want_states:
        states[:, 0] = x
    for i in range(n_steps):
        tm = i * dt + dt / 2
        act = (tm >= tf) & (tm < tc)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).to(S.rdtype)
        on = (tm >= tk) & (tm < t_off)
        Ps = Ps + dP_t * on.view(n, 1).to(S.rdtype)
        if dP_seq is not None:
            Ps = Ps + dP_seq[:, i]
        x = S.rk4_real(x, M, N, Ps, dt)
        om[:, i + 1] = x[:, 3:6]
        if want_states:
            states[:, i + 1] = x
    return om, states


def peaks_from_om(om, tk, vt, dt, grace=GRACE, device='cpu'):
    """Structural replica of verify_true.peaks_from (mask semantics identical:
    ti in float steps; grace after the decision, judged window ends at
    tk + valid_t)."""
    T = om.shape[-2] if torch.is_tensor(om) else om.shape[1]
    ti = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) * dt
    tk_t = torch.as_tensor(tk, dtype=torch.float32, device=device).unsqueeze(1)
    vt_t = torch.as_tensor(vt, dtype=torch.float32, device=device).unsqueeze(1)
    d = (om - 1.0).abs().amax(dim=2)
    m_run = ti >= (tk_t + grace)
    m_win = m_run & (ti <= tk_t + vt_t)
    return (torch.where(m_win, d, torch.full_like(d, -1.0)).amax(1),
            torch.where(m_run, d, torch.full_like(d, -1.0)).amax(1))


def run_true_batched(S, d, ks, tk, vt, dt=DT, dP=None, t_off=None,
                     dP_seq=None):
    """One scored full run per row (n = len(ks)); returns pk_win, pk_run,
    x_at (clean state at tk), om, states as numpy."""
    dev = S.device
    n = len(ks)
    Yf, dPm = fault_arrays_row(S, d, ks)
    tf_t = torch.as_tensor(np.asarray(d['t_fault'][ks]), dtype=S.rdtype,
                           device=dev)
    tc_t = torch.as_tensor(np.asarray(d['t_clear'][ks]), dtype=S.rdtype,
                           device=dev)
    tk_t = torch.as_tensor(np.asarray(tk), dtype=S.rdtype, device=dev)
    if t_off is None:                      # dp_hold = 'clear'
        t_off_t = tc_t
    else:
        t_off_t = torch.as_tensor(np.asarray(t_off), dtype=S.rdtype,
                                  device=dev)
    n_steps = int(round(float(d['t_grid'][-1]) / dt))
    dP_t = None
    if dP is not None:
        if torch.is_tensor(dP):
            dP_t = dP.to(dtype=S.rdtype, device=dev)
        else:
            dP_t = torch.as_tensor(np.asarray(dP), dtype=S.rdtype, device=dev)
    if dP_seq is not None:
        dP_seq_t = torch.as_tensor(np.asarray(dP_seq), dtype=S.rdtype,
                                   device=dev)
    else:
        dP_seq_t = None
    om_t, st_t = full_sim_np(S, tf_t, tc_t, tk_t, t_off_t, Yf, dPm, dP_t, dt,
                             n_steps, dP_seq=dP_seq_t, want_states=True)
    om = om_t.cpu().numpy()
    st = st_t.cpu().numpy()
    pk_win, pk_run = peaks_from_om(om_t, tk, vt, dt, device=dev)
    i_tk = np.round(np.asarray(tk) / dt).astype(int)
    x_at = st[np.arange(n), i_tk]
    return (pk_win.cpu().numpy(), pk_run.cpu().numpy(), x_at, om, st)


def case_flags(pk_free, pk_free_run, x_at, TK, active, limit):
    """verify_true / compare_models RunTrue conventions."""
    dev0 = np.abs(x_at[:, 3:6] - 1.0).max(axis=1)
    reach = (pk_free >= 0) & (dev0 <= limit) & active
    need = (pk_free > limit) & reach
    healthy = (pk_free >= 0) & (pk_free <= limit)
    excluded = int(np.sum((pk_free > limit) & (pk_free >= 0)
                          & (dev0 <= limit) & ~active))
    return dict(dev0=dev0, reach=reach, need=need, healthy=healthy,
                excluded=excluded)


# ── cluster bootstrap (compare_models protocol verbatim) ──────────────────────

_BOOT = {}


def boot_weights(KS_all, boot=BOOT, seed=BOOT_SEED):
    """W exactly as compare_models builds it (unique KS over the whole shared
    draw; multinomial scenario weights) so CIs are comparable across rows."""
    uniq, inv = np.unique(KS_all, return_inverse=True)
    n_sc = len(uniq)
    W = np.random.default_rng(seed).multinomial(
        n_sc, np.full(n_sc, 1.0 / n_sc), size=boot).astype(np.float64)
    _BOOT['W'] = W
    _BOOT['uniq'] = uniq
    _BOOT['inv'] = inv
    _BOOT['inv_sub'] = inv[:N_TRUE]     # sweep rows live on the true slice
    return uniq, inv


def per_sc(v):
    out = np.zeros(len(_BOOT['uniq']))
    inv = _BOOT.get('inv_used', _BOOT.get('inv_sub', _BOOT['inv']))
    np.add.at(out, inv, v)
    return out


def ci(a):
    a = a[np.isfinite(a)]
    if not len(a):
        return (float('nan'), float('nan'))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def metrics_block(per_instant, KS_all):
    """per_instant: dict name -> (numerator float array, denominator array,
    optional transform). Returns {name: {value, ci, n, n_scenarios}}."""
    res = {}
    for name, (num, den, tf) in per_instant.items():
        ns, ds = per_sc(np.asarray(num, dtype=np.float64)), \
            per_sc(np.asarray(den, dtype=np.float64))
        def est(nn, dd, tf=tf):
            with np.errstate(invalid='ignore', divide='ignore'):
                r = nn / dd
            if tf is not None:
                r = np.where(np.isfinite(r), tf(r), np.nan)
            return r
        point = float(est(ns.sum(), ds.sum())) if ds.sum() > 0 else float('nan')
        b = est(_BOOT['W'] @ ns, _BOOT['W'] @ ds)
        lo, hi = ci(b)
        n = int(round(float(np.sum(den))))
        nsc = int(np.sum(per_sc(den) > 0))
        res[name] = {'value': point, 'ci': [lo, hi], 'n': n,
                     'n_scenarios': nsc}
    return res


def rescue_metrics(flags_rows, rows_meta, dp_cols):
    """flags_rows: dict of per-instant bool arrays (need/healthy/won_t/run_ok/
    free_run_ok). rows_meta: per-instant y_type / age / dp_status. dp_cols:
    per-instant norm arrays (l2 first/max/mean, l1 mean)."""
    need, healthy = flags_rows['need'], flags_rows['healthy']
    TYP, AGE, DPS = rows_meta
    fl = 100.0
    per = {
        'rescue_true': (flags_rows['won_t'] * need, need.astype(float), None),
        'rescue_true_end_of_run': (flags_rows['run_ok'] * need,
                                   need.astype(float), None),
        'harmed_pct': (flags_rows['harmed'] * fl, healthy.astype(float), None),
        'no_dp_rescued_end_of_run':
            (flags_rows.get('free_run_ok', np.zeros(need.shape)) * need,
             need.astype(float), None),
        'dp_norm_l2_first': (dp_cols['l2_first'], np.ones(len(need)), None),
        'dp_norm_l2_max': (dp_cols['l2_max'], np.ones(len(need)), None),
        'dp_norm_l2_mean': (dp_cols['l2_mean'], np.ones(len(need)), None),
        'dp_norm_l1_mean': (dp_cols['l1_mean'], np.ones(len(need)), None),
        'dp_norm_need_l2_max': (dp_cols['l2_max'] * need,
                                need.astype(float), None),
    }
    for ti, tn in enumerate(TYPE_NAMES):
        mk = need * (TYP == ti + 1)
        per[f'rescue_true_{tn}'] = (flags_rows['won_t'] * mk,
                                    mk.astype(float), None)
    for (b0, b1), lab in zip(AGE_BINS, AGE_LABELS):
        mk = need * ((AGE >= b0) & (AGE < b1))
        per[f'rescue_true_age_{lab}'] = (flags_rows['won_t'] * mk,
                                         mk.astype(float), None)
    per['rescue_true_status3'] = (flags_rows['won_t'] * need * (DPS == 3),
                                  (need * (DPS == 3)).astype(float), None)
    per['rescue_true_status1'] = (flags_rows['won_t'] * need * (DPS == 1),
                                  (need * (DPS == 1)).astype(float), None)
    per['need_status3'] = ((DPS == 3), np.ones(len(DPS)), None)
    per['need_status2'] = ((DPS == 2), np.ones(len(DPS)), None)
    return metrics_block(per, None)


def peak_stats(om, tk, vt):
    """Mean/max true-dynamics peak |omega-1| over the judged windows."""
    pk, _ = peaks_from_om(torch.as_tensor(om), tk, vt, DT)
    pk = pk.cpu().numpy()
    fin = pk >= 0
    return {'peak_mean': float(pk[fin].mean()),
            'peak_max': float(pk.max()),
            'peak_p95': float(np.percentile(pk[fin], 95)) if fin.any()
            else float('nan')}


def koopman_closed_loop_batch(S, d, lift, K, qp, ks, seed=42, dt=DT,
                              dec_every=DEC_EVERY, hh=H):
    """Batched (over scenarios) closed-loop Koopman-MPC, Task-4 conventions:
    decisions every 20 ms over the WHOLE window from t=0, KF on measured noisy
    channels (u_applied known to the filter), QP warm per decision + sampled
    cold, |u|<=1 pu, rate 0.5/decision, held-until-clear application mask
    (dp_hold='clear' row convention: applied ΔP zeroed from the clear instant).
    Returns per-scenario om trajectories, applied-u plane, warm/cold latency
    arrays (per decision), n_dec."""
    import osqp as osqp_mod
    n = len(ks)
    A, B = lift.A, lift.B
    m = lift.n_state
    Cw = qp['Cw']; Tkk = qp['T']
    l0, u0 = qp['l0'], qp['u0']
    hinge_rows, rate_rows, term_rows = qp['hinge_rows'], qp['rate_rows'], \
        qp['term_rows']
    settings = dict(verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000,
                    polishing=True)

    # row indices for fast bound updates
    wslice = lift.omega_rows()
    rp_idx = np.zeros((hh, 3), dtype=int); rm_idx = np.zeros((hh, 3), dtype=int)
    for (kk, mm, rp, rm) in hinge_rows:
        rp_idx[kk, mm] = rp; rm_idx[kk, mm] = rm
    rp_idx = rp_idx.reshape(-1); rm_idx = rm_idx.reshape(-1)   # k-major order
    tp_idx = np.array([p for (p, mm) in term_rows])
    tm_idx = np.array([mm for (p, mm) in term_rows])
    r0_idx = [r for (c, mm, r) in rate_rows if c == 0]
    assert len(r0_idx) == 3
    oflat_p0 = l0.copy(); oflat_p1 = u0.copy()

    Yf, dPm = fault_arrays_row(S, d, ks)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, 3, 3))
    Mf, Nf = S.pe_consts(Yf)
    Tn = len(d['t_grid'])              # 1601 states, 1600 steps
    Tsteps = Tn                        # states; loop covers Tsteps-1 == 1600
    tf_np = np.asarray(d['t_fault'])[ks]; tc_np = np.asarray(d['t_clear'])[ks]
    cont = ((np.arange(Tn - 1) * dt + dt / 2)[:, None]
            < tc_np[None, :])                                      # (T-1, n)

    x = torch.cat([S.delta0, torch.ones(3), S.Pm0]) \
        .unsqueeze(0).expand(n, 9).clone()
    Pm0e = S.Pm0.unsqueeze(0).expand(n, 3)
    om = np.empty((n, Tn, 3)); om[:, 0] = x[:, 3:6].numpy()
    xhat = np.empty((n, m))
    xhat[:] = x.numpy()[:, :m]
    upl = np.zeros((n, Tn - 1, 3))
    u_prev = np.zeros((n, 3))
    m_noise = NOISE_STD[:m]
    rng = np.random.default_rng(seed)

    sols = [osqp_mod.OSQP() for _ in range(n)]
    for so in sols:
        so.setup(P=qp['P'], q=qp['q'], A=qp['A'], l=l0, u=u0, **settings)

    warm_list, cold_list = [], []
    n_dec = 0
    of1 = np.empty((n, hh, 3))
    for t in range(Tsteps - 1):
        tm = t * dt + dt / 2
        act = (tm >= tf_np) & (tm < tc_np)                     # (n,)
        act_t = torch.as_tensor(act)                            # bool (n,)
        M = torch.where(act_t.view(n, 1, 1), Mf, Mp)
        N = torch.where(act_t.view(n, 1, 1), Nf, Np)
        Pset = Pm0e + dPm * act_t.view(n, 1).to(S.rdtype)
        Uapp = upl[:, t] * cont[t][:, None]                    # numpy (n, 3)
        appl_t = torch.as_tensor(Uapp, dtype=S.rdtype)
        Pset = Pset + appl_t
        x = S.rk4_real(x, M, N, Pset, dt)
        om[:, t + 1] = x[:, 3:6].numpy()
        xnp = x.numpy()
        y = xnp[:, :m] + rng.normal(0.0, m_noise, (n, m))
        Zp = lift.phi(xhat) @ A.T + Uapp @ B.T                 # numpy f64
        xp = lift.phys_rows(Zp)                                # (n, m)
        xhat = xp + (y - xp) @ K.T
        # ── decision ────────────────────────────────────────────────────────
        if (t + 1) % dec_every == 0 and (t + 1) <= Tsteps - 1 - dec_every:
            z0 = lift.phi(xhat)
            z = z0.copy()
            for i in range(hh):
                z = z @ A.T
                of1[:, i] = z[:, wslice] - 1.0
            of1f = of1.reshape(n, hh * 3)
            ofH = of1[:, hh - 1, :]
            di = t // dec_every
            cold_this = (di % COLD_DEC == 0)
            if len(warm_list) <= di:
                warm_list.append(np.empty(n))
                cold_list.append(np.full(n, np.nan))
            for j in range(n):
                lu = oflat_p0.copy(); uu = oflat_p1.copy()
                lu[rp_idx] = of1f[j] - QP_DB
                lu[rm_idx] = -of1f[j] - QP_DB
                lu[tp_idx] = ofH[j]; lu[tm_idx] = -ofH[j]
                for mm in range(3):
                    lu[r0_idx[mm]] = -QP_RATE + u_prev[j][mm]
                    uu[r0_idx[mm]] = QP_RATE + u_prev[j][mm]
                tw = time.perf_counter()
                sols[j].update(l=lu, u=uu)
                res = sols[j].solve()
                warm_list[di][j] = (time.perf_counter() - tw) * 1e3
                ucell = np.clip(res.x[:3 * N_DEC].reshape(N_DEC, 3)[0],
                                -QP_UMAX - 1e-9, QP_UMAX + 1e-9).copy()
                u_prev[j] = ucell
                upl[j, t + 1:t + 1 + dec_every] = ucell
                if cold_this and (j % COLD_SC == 0):
                    tcold = time.perf_counter()
                    s2 = osqp_mod.OSQP()
                    s2.setup(P=qp['P'], q=qp['q'], A=qp['A'], l=lu, u=uu,
                             **settings)
                    s2.solve()
                    cold_list[di][j] = (time.perf_counter() - tcold) * 1e3
            n_dec += 1
    warm = np.stack(warm_list, axis=1); cold = np.stack(cold_list, axis=1)
    return om, upl, cont, warm, cold, n_dec


def win_flags_on_row(om_row, tk, vt, dt=DT, grace=GRACE):
    """om_row: (T, 3); tk/vt: (n_inst,) — vectorized window peaks for all
    instants sharing one trajectory."""
    T = om_row.shape[0]
    times = np.arange(T) * dt
    d = np.abs(om_row[:, :] - 1.0).max(axis=1)                 # (T,)
    in_run = times[:, None] >= (np.asarray(tk)[None, :] + grace)
    in_win = in_run & (times[:, None] <= np.asarray(tk)[None, :] + vt[None, :])
    pk_win = np.where(in_win, d[:, None], -1.0).max(axis=0)
    pk_run = np.where(in_run, d[:, None], -1.0).max(axis=0)
    return pk_win, pk_run


def dp_cols_from_plane(upl_row, tk, t_off):
    """Per-instant ‖ΔP‖ series from a scenario's applied-u plane.
    upl_row: (T-1, 3); tk: (n_inst,); t_off: clear time per instant."""
    upl_row = np.asarray(upl_row)
    appl = upl_row                                             # (T-1, 3)
    inst_tk = np.searchsorted(np.arange(appl.shape[0]) * DT + DT / 2,
                              np.asarray(tk), side='left')
    n = len(tk)
    out = {kk: np.zeros(n) for kk in
           ['l2_first', 'l2_max', 'l2_mean', 'l1_mean']}
    for j in range(n):
        a = inst_tk[j]
        bb = np.searchsorted(np.arange(appl.shape[0]) * DT + DT / 2,
                             t_off[j], side='left')
        rows = appl[a:max(a, bb)]
        if len(rows):
            n2 = np.linalg.norm(rows, axis=1)
            out['l2_first'][j] = np.linalg.norm(rows[0])
            out['l2_max'][j] = n2.max()
            out['l2_mean'][j] = n2.mean()
            out['l1_mean'][j] = np.abs(rows).sum(1).mean()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Task 5 — cross-check against verify_true.py itself
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cmd_crosscheck(args, out_dir):
    t0 = time.time()
    ckpt = os.path.join(BUS9, 'receding_cascade/refined/best.pt') \
        if args.ckpt is None else os.path.join(BUS9, args.ckpt)
    KS_all, TK_all, inst = load_instants(INSTANTS_DEFAULT)
    assert inst['split'] == args.split
    N = args.n
    KS, TK = KS_all[:N], TK_all[:N]
    vtm = import_verify_true(ckpt, out_dir)   # reference machinery (f32 CUDA)
    LIMIT = vtm.model.limit
    assert abs(float(LIMIT) - OMEGA_LIMIT) < 1e-12
    sub = run_verify_subprocess(out_dir, N, split=args.split)
    d = load_data()
    A, _ck = ckpt_args(ckpt)
    active, vt_arr = build_windows(A, d, KS, TK)

    # ── shared emitted ΔP: the official machinery's o['dP'] ─────────────────
    b0 = vtm.make_batch(KS, np.random.default_rng(11), tk_fixed=TK)
    o = vtm.model(b0[0].to(vtm.device), b0[1].to(vtm.device), hard=True,
                  ctrl=True, vmask=b0[3].to(vtm.device))
    dP = o['dP']
    pk_m, has_m = vtm.model.judged(o['w_ctrl'], b0[3].to(vtm.device))
    print(f'emitted dP (shared): shape {tuple(dP.shape)}, '
          f'norm2 mean {dP.pow(2).sum(1).sqrt().mean().item():.6f}')

    # ── reference path: verify_true's own functions, VERBATIM ────────────────
    tr_free_ref, st_free_ref = vtm.full_sim(KS, TK, None)
    tr_ctrl_ref, _ = vtm.full_sim(KS, TK, dP, hold='clear', vt=vt_arr)
    pf_ref, pfr_ref = vtm.peaks_from(tr_free_ref, TK, vt_arr)
    pc_ref, pcr_ref = vtm.peaks_from(tr_ctrl_ref, TK, vt_arr)

    # ── my scorer, f32 CUDA, INDEPENDENT network construction ────────────────
    S32 = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
                os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
                dtype=torch.float32, device='cuda', lines=d['lines'])
    pk_f32, pkr_f32, x_at32, om32, st32 = run_true_batched(
        S32, d, KS, TK, vt_arr)
    pk_c32, pkr_c32, _, _, _ = run_true_batched(
        S32, d, KS, TK, vt_arr, dP=dP)
    # (a1) integration-loop equality with the SAME networks: my loop feeding
    #      vt._full_sim_ref the same tensors (dP=None path = free run)
    tf_t = torch.as_tensor(np.asarray(d['t_fault'][KS]), dtype=torch.float32,
                           device=vtm.device)
    tc_t = torch.as_tensor(np.asarray(d['t_clear'][KS]), dtype=torch.float32,
                           device=vtm.device)
    tk_t = torch.as_tensor(TK, dtype=torch.float32, device=vtm.device)
    Yf, dPm = fault_arrays_row(S32, d, KS)
    om_loop, _ = full_sim_np(S32, tf_t, tc_t, tk_t, tc_t, Yf, dPm,
                             torch.zeros(len(KS), 3, device=vtm.device), DT,
                             1600, want_states=False)
    res_a1_free = float(np.abs(tr_free_ref.cpu().numpy()
                               - om_loop.cpu().numpy()).max())
    # ctrl with vt's own networks through MY loop:
    tr_ctrl_mine_loop, _ = full_sim_np(S32, tf_t, tc_t, tk_t, tc_t, Yf, dPm,
                                       dP, DT, 1600, want_states=False)
    res_a1_ctrl = float(np.abs(tr_ctrl_ref.cpu().numpy()
                               - tr_ctrl_mine_loop.cpu().numpy()).max())

    # (d) sequence-mode license: constant ΔP emitted as a per-decision
    #     sequence; step indices from the SAME midpoint predicates the const
    #     path's on-mask uses (tm = i*dt + dt/2)
    mid = np.arange(1600) * DT + DT / 2
    i_tk32 = np.searchsorted(mid, TK, side='left')       # first i: tm >= tk
    i_off32 = np.searchsorted(mid, d['t_clear'][KS], side='left')
    seq = torch.zeros(len(KS), 1600, 3, dtype=torch.float32,
                      device=vtm.device)
    for j in range(len(KS)):
        a, bb = i_tk32[j], i_off32[j]
        seq[j, a:bb] = dP[j]
    om_seq, _ = full_sim_np(S32, tf_t, tc_t, tk_t, tc_t, Yf, dPm,
                            torch.zeros(len(KS), 3, device=vtm.device), DT,
                            1600, dP_seq=seq, want_states=False)
    res_d = float(np.abs(om_seq.cpu().numpy()
                         - tr_ctrl_mine_loop.cpu().numpy()).max())

    # my-flag aggregates (used in level b) — from MY f32 run
    fl = case_flags(pk_f32, pkr_f32, x_at32, TK, active, LIMIT)
    win_t = pk_c32 <= LIMIT
    win_run = pkr_c32 <= LIMIT
    agg_mine = dict(
        n=int(active.size), need=int(fl['need'].sum()),
        resc_true=int(win_t[fl['need']].sum()),
        ok_true=int(win_t[fl['reach']].sum()), ok_n=int(fl['reach'].sum()),
        resc_run=int(win_run[fl['need']].sum()),
        norm2_mean=float(dP.pow(2).sum(1).sqrt().mean().item()))
    free_run_ok = (pkr_f32 <= LIMIT)
    agg_mine['free_run'] = int(free_run_ok[fl['need']].sum())
    agg_mine['resc_model'] = int((pk_m <= LIMIT)[fl['need']].sum())

    # (b) my f32 aggregates vs the official subprocess JSON
    ref = sub
    def ratio(a, b): return (a / b if b else float('nan'))
    checks_b = {
        'rescue_true':      [ratio(agg_mine['resc_true'], agg_mine['need']),
                             ref['rescue_true']],
        'rescue_true_full_run': [ratio(agg_mine['resc_run'],
                                       agg_mine['need']),
                                 ref['rescue_true_full_run']],
        'no_dp_full_run':   [ratio(agg_mine['free_run'], agg_mine['need']),
                             ref['no_dp_full_run']],
        'ok_true':          [ratio(agg_mine['ok_true'], agg_mine['ok_n']),
                             ref['ok_true']],
        'n_need':           [agg_mine['need'], ref['n_need']],
        'excluded':         [fl['excluded'],
                             ref['excluded_no_active_fault']],
        'mean_norm2_dP':    [agg_mine['norm2_mean'], float(ref['mean_dP'])],
        'rescue_model':     [ratio(agg_mine['resc_model'], agg_mine['need']),
                             ref['rescue_model']],
    }
    res_b = {}
    for kk, (mine, refv) in checks_b.items():
        ab = 0.0 if (not np.isfinite(mine) and not np.isfinite(refv)) \
            else float(abs(mine - refv))
        res_b[kk] = {'mine': float(mine) if np.isfinite(mine) else None,
                     'verify_true': float(refv) if np.isfinite(refv) else None,
                     'absdiff': ab}
    b_max = max((v['absdiff'] for v in res_b.values() if v['absdiff']
                 == v['absdiff']), default=0.0)

    # (a2) end-to-end f32 peaks: my scorer vs vt path
    def t(x):
        return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    res_a2 = {
        'peak_free_win_max_absdiff': float(np.abs(t(pk_f32)
                                                  - t(pf_ref)).max()),
        'peak_ctrl_win_max_absdiff': float(np.abs(t(pk_c32)
                                                  - t(pc_ref)).max()),
        'peak_free_run_max_absdiff': float(np.abs(t(pkr_f32)
                                                  - t(pfr_ref)).max()),
        'peak_ctrl_run_max_absdiff': float(np.abs(t(pkr_c32)
                                                  - t(pcr_ref)).max()),
        'x_at_max_absdiff': float(np.abs(
            x_at32 - np.asarray(t(st_free_ref))[
                np.arange(N), np.round(TK / DT).astype(int)]).max()),
    }

    # (c) informational: f64-CPU scorer vs the vt f32 path
    S64 = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
                os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
                dtype=torch.float64, lines=d['lines'])
    pk_f64, pkr_f64, x_at64, _, _ = run_true_batched(S64, d, KS, TK, vt_arr)
    pk_c64, pkr_c64, _, _, _ = run_true_batched(S64, d, KS, TK, vt_arr, dP=dP)
    res_c = {
        'peak_free_win_max_absdiff': float(np.abs(pk_f64 - t(pf_ref)).max()),
        'peak_ctrl_win_max_absdiff': float(np.abs(pk_c64 - t(pc_ref)).max()),
        'peak_free_run_max_absdiff': float(np.abs(pkr_f64
                                                  - t(pfr_ref)).max()),
        'peak_ctrl_run_max_absdiff': float(np.abs(pkr_c64
                                                  - t(pcr_ref)).max()),
    }

    # per-instant table for the record
    per_instant = []
    for j in range(N):
        per_instant.append({
            'sl_i': j, 'k': int(KS[j]), 'tk': float(TK[j]),
            'valid_t': float(vt_arr[j]), 'active': bool(active[j]),
            'dp_norm2': float(torch.linalg.norm(dP[j]).item()),
            'peak_free_ref': float(t(pf_ref)[j]),
            'peak_free_mine32': float(pk_f32[j]),
            'peak_free_mine64': float(pk_f64[j]),
            'peak_ctrl_ref': float(t(pc_ref)[j]),
            'peak_ctrl_mine32': float(pk_c32[j]),
            'peak_ctrl_mine64': float(pk_c64[j]),
            'peak_free_run_mine64': float(pkr_f64[j]),
            'need_mine': bool(fl['need'][j]),
        })

    # GATE: (a1) ~0 + (a2) ~1e-6 + (b) ratios/counts exact-class + (d) ~0
    gate = {'a1_free': res_a1_free, 'a1_ctrl': res_a1_ctrl, 'a2': res_a2,
            'b': res_b, 'c': res_c, 'd_seq': res_d}
    a2_max = max(res_a2.values())
    verdict = (res_a1_free < 1e-6 and res_a1_ctrl < 1e-6 and
               a2_max < 1e-6 and b_max < 1e-6 and res_d < 1e-6)
    art = {
        'cmd': ' '.join(sys.argv), 'ckpt': ckpt,
        'instants': {'file': INSTANTS_DEFAULT, 'n': N, 'seed': 42,
                     'split': args.split,
                     'note': 'first N of the shared seed-42 draw '
                             '(compare slice)'},
        'shared_dP_norm2': [float(torch.linalg.norm(dP[j]).item())
                            for j in range(N)],
        'gate': gate, 'GATE_PASS': bool(verdict),
        'per_instant': per_instant,
        'runtime_s': time.time() - t0,
    }
    p = os.path.join(out_dir, 'task5_crosscheck.json')
    json.dump(art, open(p, 'w'), indent=1, default=float)
    print(f'saved {p}')
    print('GATE (a1 free/ctrl):', res_a1_free, res_a1_ctrl)
    print('GATE (a2 end-to-end f32):', json.dumps(res_a2))
    print('GATE (b aggregates):', json.dumps(res_b))
    print('GATE (d seq-license):', res_d)
    print('INFO (c f64-vs-f32):', json.dumps(res_c))
    print('VERDICT:', 'PASS' if verdict else 'FAIL — STOP, diagnose')
    if not verdict:
        sys.exit(2)


# ══════════════════════════════════════════════════════════════════════════════
# Task 6 — full seed-42 validation sweep (row-selectable, subset-capable)
# ══════════════════════════════════════════════════════════════════════════════

INK, INK2, GRID = '#2c2c2a', '#6b6a64', '#e4e3dd'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#8959a8']
ROW_ALIAS = {'koopman_gov': 'koopman_gov_fullchan',
             'korda_native': 'korda_native_bare'}


def assemble_head_rows():
    """Head columns AS-IS from the canonical seed-42 artifacts (never re-run)."""
    crp = os.path.join(COMP, 'compare_results.json')
    cr = json.load(open(crp))
    prov = {
        'compare_results.json': sha256_of(crp),
        'instants.json': sha256_of(INSTANTS_DEFAULT),
        'shared_verify_dirs': ['shared_LSTMref_s42/verify_results.json',
                               'shared_MLPref_s42/verify_results.json'],
        'profile_latency': ['bus_9/profile_latency_out/profile_latency.json',
                            'wiki concepts/decision-latency-profile.md: '
                            'eager 464.9 ms @256 / graph 126.6 ms @256; '
                            '447.6 / 138.6 ms at bs=1'],
        'note': 'head rows NOT re-run; per-instant true-dynamics peak |w-1| '
                'is not stored in the canonical artifacts -> marked n/a; '
                'dp_norm_need_l2_max / dp_norm_l2_mean aliases of the head '
                'dP_norm_need / dP_norm metrics (constant held dP).'}
    rows = {}
    for label, mlabel in [('head_lstm_cascade', 'LSTM+LSTM'),
                          ('head_mlp', 'MLP+MLP')]:
        M = json.loads(json.dumps(cr['models'][mlabel]['metrics']))
        M['dp_norm_need_l2_max'] = M['dP_norm_need']
        M['dp_norm_l2_mean'] = M['dP_norm']
        rows[label] = dict(
            kind='head', mlabel=mlabel, ckpt_path=cr['models'][mlabel]['path'],
            metrics=M,
            latency=dict(
                eager_single_ms=cr['models'][mlabel]['latency_ms']['single']
                ['decision'],
                eager_batch_ms=cr['models'][mlabel]['latency_ms']['batch']
                ['decision'],
                graphed_wiki_ms=126.6, graphed_wiki_bs1_ms=138.6,
                source='compare_out_s42 (eager, measured) + profile_latency '
                       'wiki (graphed canonical)'),
            provenance=prov,
            true_peak='n/a (not stored in canonical artifacts; not re-run '
                      'per plan)')
    return rows


def figure_quality(table, png):
    """Quality bars with cluster-bootstrap CIs; compare-scripts styling."""
    from collections import OrderedDict
    ORDER = ['head_lstm_cascade', 'head_mlp', 'koopman_gov_fullchan',
             'korda_native_bare', 'scheduled_scan', 'oracle_label',
             'no_dp_floor']
    ORDER = [r for r in ORDER
             if (r != 'scheduled_scan') or ('scheduled_scan' in table['rows'])]
    SHORT = OrderedDict([('head_lstm_cascade', 'HEAD\nLSTM'),
                         ('head_mlp', 'HEAD\nMLP'),
                         ('koopman_gov_fullchan', 'KOOPMAN\ngov\nfullchan'),
                         ('korda_native_bare', 'KORDA\nnative\nbare'),
                         ('scheduled_scan', 'SCHED\nscan'),
                         ('oracle_label', 'ORACLE\nlabel'),
                         ('no_dp_floor', 'NO ΔP\nfloor')])
    panels = [('rescue_true', 'Rescue rate, TRUE dynamics\n(judged window)'),
              ('rescue_true_end_of_run', 'Rescue to END OF RUN\n(needy cases)'),
              ('harmed_pct', 'HARMED [% of healthy]\n(run continuously)'),
              ('dp_norm_l2_mean', '‖ΔP‖₂, per-instant mean [pu]')]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    for ax, (key, title) in zip(axes, panels):
        vals, los, his = [], [], []
        for r in ORDER:
            m = table['rows'][r]['metrics'].get(key)
            if m is None or not np.isfinite(m.get('value', np.nan)):
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
            else:
                vals.append(m['value']); los.append(m['ci'][0])
                his.append(m['ci'][1])
        x = np.arange(len(ORDER))
        ax.bar(x, vals, width=0.62, color=COL[:len(ORDER)], zorder=3)
        for xi, v, lo, hi in zip(x, vals, los, his):
            if np.isfinite(v) and np.isfinite(lo):
                ax.errorbar(xi, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                            fmt='none', color=INK, elinewidth=1.2, capsize=3,
                            zorder=5)
                lab = f'{v:.3f}' if abs(v) < 2 else f'{v:.1f}'
                ax.text(xi, max(v, hi), lab, ha='center', va='bottom',
                        fontsize=7.5, color=INK,
                        bbox=dict(boxstyle='square,pad=0.12', fc='white',
                                  ec='none'))
        ax.set_xticks(x, [SHORT[r] for r in ORDER], fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.tick_params(axis='y', labelsize=7.5, colors=INK2)
        ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
    fig.suptitle(f"9-bus Koopman-MPC baseline sweep — seed-42 shared draw "
                 f"(test, {table['meta']['n_true_instants']} instants / "
                 f"{table['meta'].get('n_scenarios_true_subset', '?')} "
                 f"scenarios; bars = 95% cluster-bootstrap CI; head rows from "
                 f"canonical seed-42 artifacts, not re-run)", fontsize=10,
                 color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print('->', png)


def figure_latency(table, png):
    lines = [
        ('head LSTM (cascade)',
         'eager %.0f ms (single) / %.0f (batch) — graphed canonical 126.6 '
         '(bs 256), 138.6 (bs 1)' % (
             table['rows']['head_lstm_cascade']['latency']['eager_single_ms'],
             table['rows']['head_lstm_cascade']['latency']['eager_batch_ms'])),
        ('head MLP',
         'eager %.0f ms (single) / %.0f (batch) — graphed canonical 126.6' % (
             table['rows']['head_mlp']['latency']['eager_single_ms'],
             table['rows']['head_mlp']['latency']['eager_batch_ms'])),
        ('oracle / no-ΔP floor', 'no decision (label / zero action)'),
    ]
    for vtag in ('koopman_gov_fullchan', 'korda_native_bare'):
        if vtag in table['rows'] and 'latency' in table['rows'][vtag]:
            lat = table['rows'][vtag]['latency']
            lines.append((vtag.replace('fullchan', 'gov_fullchan').replace(
                'bare', 'bare') + ' (QP)',
                'warm %.2f / cold %.2f ms (median, fresh per-decision; %d '
                'decisions, %d cold samples)' % (
                    lat['warm_median_ms'], lat['cold_median_ms'],
                    lat['n_decisions_per_scenario'],
                    lat['n_cold_samples'])))
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axis('off')
    tb = ax.table(cellText=[[a, b] for a, b in lines],
                  colLabels=['row', 'decision latency [ms]'], loc='center',
                  cellLoc='left', colWidths=[0.28, 0.72])
    tb.auto_set_font_size(False); tb.set_fontsize(9)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor('#e4e3dd')
            cell.set_text_props(color=INK, weight='bold')
        else:
            cell.set_text_props(color=INK)
        cell.set_edgecolor(GRID)
    ax.set_title('Decision latency by row (9-bus, seed-42 shared draw; head '
                 'latency cited from canonical artifacts, not re-measured)',
                 fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print('->', png)


def label_design_check(S64, d, md, ksid):
    """Oracle-machinery validation against the LABEL's own documented design:
    y_delta_p applied [t_apply, t_remove] (held-to-clear), judged from
    t_apply + 0.2 s (DATASET.md / compute_delta_p) -> peak must reproduce
    best_maxdev_window (their RK4 h=0.01 s scan; ours 5 ms) and every
    dp_status==1 scenario must sit <= omega_limit. This is the scorer's
    provenance tie to the dataset labels, independent of the head machinery."""
    t_apply = np.asarray(md['t_apply'])[ksid]
    t_rm = np.asarray(md['t_remove'])[ksid]
    ydp = np.asarray(md['y_delta_p'])[ksid]
    best = np.asarray(md['best_maxdev_window'])[ksid]
    status = np.asarray(md['dp_status'])[ksid]
    vt_d = t_rm - t_apply
    pk_win, pk_run, _, _, _ = run_true_batched(S64, d, ksid, t_apply, vt_d,
                                               dP=ydp, t_off=t_rm)
    pk_win = np.where(pk_win >= 0, pk_win, pk_run)   # window < grace: use run
    diff_abs = np.abs(np.maximum(pk_win, pk_run) - best)   # judged to run end
    fin = np.isfinite(diff_abs)
    s1 = status == 1; s3 = status == 3; s0 = status == 0
    def frac(mask, bound):
        return float(np.mean(np.maximum(pk_win, pk_run)[mask] <= bound)) \
            if mask.any() else None
    return {
        'n': int(len(ksid)),
        'max_absdiff_vs_best_maxdev_window': float(diff_abs[fin].max()),
        'median_absdiff': float(np.median(diff_abs[fin])),
        'n_finite_pairs': int(fin.sum()),
        'n_gt_1e-3': int((diff_abs[fin] > 1e-3).sum()),
        'n_gt_1e-5': int((diff_abs[fin] > 1e-5).sum()),
        'design_peak_le_limit_minus_margin_frac': {
            'status1': frac(s1, 0.005 - 1e-4 + 3 * 5e-4),
            'status3': frac(s3, 0.005 - 1e-4 + 3 * 5e-4),
            'status0': frac(s0, 0.005)},
        'status1_design_rescue_le_limit_frac': frac(s1, 0.005),
        'note': 'judged from t_apply+0.2 to RUN END (the stored window: '
                'short faults peak after reconnection), our 5 ms integrator '
                'vs the label scan stored best_maxdev_window (their RK4 '
                'h=0.01 s): expect ~1e-4-class dt agreement; status-1 must '
                'sit at/below the 0.005-1e-4 design point',
    }


def load_or_load_cache(npz_path, keys, cache_default=True):
    """Cache lookup in --out then the shipped default out dir."""
    p = npz_path if os.path.exists(npz_path) else \
        os.path.join(DEFAULT_OUT, os.path.basename(npz_path))
    if cache_default and os.path.exists(p):
        return np.load(p), p
    return None, None


@torch.no_grad()
def cmd_sweep(args, out_dir):
    t0 = time.time()
    ckpt = os.path.join(BUS9, args.ckpt)
    A, _ck = ckpt_args(ckpt)
    KS_all, TK_all, inst = load_instants(INSTANTS_DEFAULT)
    md = load_meta()
    assert abs(float(md['omega_limit']) - OMEGA_LIMIT) < 1e-12
    d = load_data()
    LIMIT = float(md['omega_limit'])
    subset = args.n is not None and args.n < N_TRUE
    n = args.n if args.n is not None else N_TRUE
    KS, TK = KS_all[:n], TK_all[:n]
    y_type = d['y_type'][KS]
    y_age = TK - d['t_fault'][KS]
    DPS = np.asarray(md['dp_status'])[KS]
    ydp = np.asarray(md['y_delta_p'])[KS]              # (n, 3) label
    active, vt = build_windows(A, d, KS, TK)
    boot_weights(KS_all)
    _BOOT['inv_used'] = _BOOT['inv'][:n]     # subset-aware bootstrap mapping
    print(f'true subset: {len(KS)} instants, {len(np.unique(KS))} scenarios'
          f"  (rows: {args.row})", flush=True)

    S64 = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
                os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
                dtype=torch.float64, lines=d['lines'])

    # row selection
    sel = {'floor', 'oracle', 'koopman_gov_fullchan', 'korda_native_bare'} \
        if args.row == 'all' else \
        {ROW_ALIAS.get(args.row, args.row)}

    # ── floor (no ΔP) — needed for need/healthy flags of every row ──────────
    cache_fo = os.path.join(out_dir, '_sweep_floor_oracle_cache.npz')
    def compute_floor():
        pk_free = np.empty(len(KS)); pk_free_run = np.empty(len(KS))
        x_at_all = np.empty((len(KS), 9))
        for s in range(0, len(KS), CHUNK):
            sl = slice(s, min(s + CHUNK, len(KS)))
            pw, pr, xa, _, _ = run_true_batched(S64, d, KS[sl], TK[sl],
                                                vt[sl])
            pk_free[sl] = pw; pk_free_run[sl] = pr; x_at_all[sl] = xa
            print(f'  floor chunk {sl.stop}/{len(KS)}', flush=True)
        return pk_free, pk_free_run, x_at_all

    z_fo, cache_fo_path = load_or_load_cache(
        cache_fo, ('pk_free', 'pk_free_run', 'x_at'), args.cache)
    cache_used = z_fo is not None and not subset
    if cache_used:
        pk_free = z_fo['pk_free'][:n]
        pk_free_run = z_fo['pk_free_run'][:n]
        x_at_all = z_fo['x_at'][:n]
        print('floor: loaded from cache', cache_fo_path, flush=True)
    else:
        pk_free, pk_free_run, x_at_all = compute_floor()
        if not subset:
            np.savez_compressed(cache_fo, pk_free=pk_free,
                                pk_free_run=pk_free_run, x_at=x_at_all)
    fl_floor = case_flags(pk_free, pk_free_run, x_at_all, TK, active, LIMIT)
    need = fl_floor['need']; healthy = fl_floor['healthy']
    free_run_ok = (pk_free_run <= LIMIT)
    print(f'need (floor): {int(need.sum())} / {len(KS)}; excluded no-active: '
          f'{fl_floor["excluded"]}', flush=True)
    floor_drift = None
    if subset and z_fo is not None:
        # subset verification: fresh subset scorer vs the shipped per-instant
        # floor cache on those instants
        floor_drift = {
            'pk_free_max_absdiff': float(np.abs(pk_free
                                                - z_fo['pk_free'][:n]).max()),
            'pk_free_run_max_absdiff': float(np.abs(
                pk_free_run - z_fo['pk_free_run'][:n]).max())}
        print('floor cache drift (subset):', floor_drift, flush=True)
    rows_meta = (y_type, y_age, DPS)

    def flags_for(pk_c, pk_c_run):
        return dict(need=need, healthy=healthy, reach=fl_floor['reach'],
                    won_t=pk_c <= LIMIT, run_ok=pk_c_run <= LIMIT,
                    free_run_ok=free_run_ok,
                    harmed=healthy & (pk_c > LIMIT))

    def row_metrics(pk_c, pk_c_run, dp_map):
        flags = flags_for(pk_c, pk_c_run)
        met = rescue_metrics(flags, rows_meta, dp_map)
        fin = pk_c >= 0
        met['peak_abs_omega_judged_mean'] = {
            'value': float(pk_c[fin].mean()) if fin.any() else float('nan'),
            'ci': None, 'n': int(fin.sum()), 'n_scenarios': None}
        met['peak_abs_omega_all_max'] = {
            'value': float(pk_c.max()), 'ci': None, 'n': None,
            'n_scenarios': None}
        return flags, met

    # ── floor row ────────────────────────────────────────────────────────────
    rows_out = assemble_head_rows()

    def floor_row_metrics():
        met = rescue_metrics(
            dict(need=need, healthy=healthy, reach=fl_floor['reach'],
                 won_t=pk_free <= LIMIT, run_ok=free_run_ok,
                 free_run_ok=free_run_ok, harmed=np.zeros(len(KS), bool)),
            rows_meta,
            {'l2_first': np.zeros(len(KS)), 'l2_max': np.zeros(len(KS)),
             'l2_mean': np.zeros(len(KS)), 'l1_mean': np.zeros(len(KS))})
        fin = pk_free >= 0
        met['peak_abs_omega_judged_mean'] = {
            'value': float(pk_free[fin].mean()) if fin.any()
            else float('nan'), 'ci': None, 'n': int(fin.sum()),
            'n_scenarios': None}
        met['peak_abs_omega_all_max'] = {'value': float(pk_free.max()),
                                         'ci': None, 'n': None,
                                         'n_scenarios': None}
        return met
    if 'floor' in sel:
        met_floor = floor_row_metrics()
        met_floor['note'] = 'no-ΔP floor: rescue in-window is 0 by the need ' \
                            'definition; end-of-run rescue = free-run recovery'
        rows_out['no_dp_floor'] = dict(kind='sweep', metrics=met_floor)

    # ── oracle row (label ΔP held-to-clear) ─────────────────────────────────
    cache_or = os.path.join(out_dir, '_sweep_oracle_cache.npz')
    oracle_drift = None
    def compute_oracle():
        pk_or = np.empty(len(KS)); pk_or_run = np.empty(len(KS))
        for s in range(0, len(KS), CHUNK):
            sl = slice(s, min(s + CHUNK, len(KS)))
            pw, pr, _, _, _ = run_true_batched(S64, d, KS[sl], TK[sl],
                                               vt[sl], dP=ydp[sl])
            pk_or[sl] = pw; pk_or_run[sl] = pr
            print(f'  oracle chunk {sl.stop}/{len(KS)}', flush=True)
        return pk_or, pk_or_run

    z_or, cache_or_path = load_or_load_cache(
        cache_or, ('pk_or', 'pk_or_run'), args.cache)
    cache_used_or = z_or is not None and not subset
    if 'oracle' in sel:
        if cache_used_or:
            pk_or = z_or['pk_or'][:n]; pk_or_run = z_or['pk_or_run'][:n]
            print('oracle: loaded from cache', cache_or_path, flush=True)
        else:
            pk_or, pk_or_run = compute_oracle()
            if not subset:
                np.savez_compressed(cache_or, pk_or=pk_or, pk_or_run=pk_or_run)
        if subset and z_or is not None:
            # subset verification: fresh subset scorer vs the shipped per-
            # instant cache on those instants
            oracle_drift = {
                'pk_or_max_absdiff': float(np.abs(pk_or
                                                  - z_or['pk_or'][:n]).max()),
                'pk_or_run_max_absdiff': float(np.abs(
                    pk_or_run - z_or['pk_or_run'][:n]).max())}
            print('oracle cache drift (subset):', oracle_drift, flush=True)
        flags_or, met_or = row_metrics(pk_or, pk_or_run, {
            'l2_first': np.linalg.norm(ydp, axis=1),
            'l2_max': np.linalg.norm(ydp, axis=1),
            'l2_mean': np.linalg.norm(ydp, axis=1),
            'l1_mean': np.abs(ydp).sum(1)})
        met_or['note'] = 'oracle: label y_delta_p applied from tk, held to ' \
                         'clear (dp_hold=clear), TRUE dynamics, f64 CPU ' \
                         'scorer'
        rows_out['oracle_label'] = dict(kind='sweep', metrics=met_or)

    # ── koopman rows ─────────────────────────────────────────────────────────
    koop_drift = {}
    for vtag in [r for r in ('koopman_gov_fullchan', 'korda_native_bare')
                 if r in sel]:
        import osqp as osqp_mod  # noqa: F401  (solver availability check)
        t_v = time.time()
        if vtag == 'koopman_gov_fullchan':
            lift = GovLift(S64, out_dir); m = 9
        else:
            lift = BareLift(out_dir); m = 6
        K = make_kf(lift, m, S64)['K']
        qp = build_qp_layout(lift, lift.A, lift.B)
        uk = np.unique(KS)
        cache_k = os.path.join(out_dir, f'_sweep_cl_{vtag}.npz')

        def eval_rows(omK, upl):
            """Window peaks + ‖ΔP‖ cols on the CURRENT instants from
            per-scenario closed-loop trajectories (omK/upl row i = uk[i])."""
            pk_c = np.empty(len(KS)); pk_c_run = np.empty(len(KS))
            dpcols = {k2: np.zeros(len(KS)) for k2 in
                      ['l2_first', 'l2_max', 'l2_mean', 'l1_mean']}
            pos = {int(g): gi for gi, g in enumerate(uk)}
            for g in uk:
                mask = KS == g
                pw, pr = win_flags_on_row(omK[pos[int(g)]], TK[mask],
                                          vt[mask])
                pk_c[mask] = pw; pk_c_run[mask] = pr
                cols = dp_cols_from_plane(
                    upl[pos[int(g)]], TK[mask],
                    np.full(int(mask.sum()), float(d['t_clear'][g])))
                for k2 in dpcols:
                    dpcols[k2][mask] = cols[k2]
            return pk_c, pk_c_run, dpcols

        cache_used_k = False
        if getattr(args, 'cache', True) and not subset:
            zk, ckp = load_or_load_cache(cache_k, (), args.cache)
            if zk is not None and zk['omK'].shape[0] == len(uk):
                omK, upl = zk['omK'], zk['upl']
                warm, cold = zk['warm'], zk['cold']
                n_dec = int(zk['n_dec'])
                cache_used_k = True
                print(f'[{vtag}] closed loop loaded from cache {ckp}',
                      flush=True)
        if not cache_used_k:
            # subset mode also consults the SHIPPED full-draw cache for a
            # cache-consistent evaluation (omK rows map by scenario id);
            # fresh closed-loop below provides the drift report.
            omK = upl = warm = cold = None; n_dec = None
        if cache_used_k:
            pk_c, pk_c_run, dpcols = eval_rows(omK, upl)
        else:
            omK, upl, cont, warm, cold, n_dec = koopman_closed_loop_batch(
                S64, d, lift, K, qp, uk, seed=42)
            pk_c, pk_c_run, dpcols = eval_rows(omK, upl)
            if subset:
                np_path = os.path.join(out_dir, f'_sweep_cl_{vtag}.npz')
                np.savez_compressed(np_path, omK=omK, upl=upl, warm=warm,
                                    cold=cold, n_dec=n_dec)
            else:
                np.savez_compressed(cache_k, omK=omK, upl=upl, warm=warm,
                                    cold=cold, n_dec=n_dec)
        if subset:
            # drift report: fresh-subset closed-loop vs the shipped full-run
            # cache (expected small: batched measurement-noise stream differs)
            cache_k_full = os.path.join(DEFAULT_OUT, f'_sweep_cl_{vtag}.npz')
            zk = np.load(cache_k_full) if os.path.exists(cache_k_full) else None
            ckp = cache_k_full
            if zk is not None and zk['omK'].shape[0] == \
                    len(np.unique(KS_all[:N_TRUE])):
                uk_full = np.unique(KS_all[:N_TRUE])
                pos_full = {int(g): gi for gi, g in enumerate(uk_full)}
                pkf = np.empty(len(KS)); pkrf = np.empty(len(KS))
                for gi, g in enumerate(uk):
                    mask = KS == g
                    pw, pr = win_flags_on_row(
                        zk['omK'][pos_full[int(g)]], TK[mask], vt[mask])
                    pkf[mask] = pw; pkrf[mask] = pr
                koop_drift[vtag] = {
                    'cache': os.path.basename(ckp),
                    'pk_win_max_absdiff_fresh_vs_cache':
                        float(np.abs(pk_c - pkf).max()),
                    'pk_run_max_absdiff_fresh_vs_cache':
                        float(np.abs(pk_c_run - pkrf).max()),
                    'note': 'fresh subset closed-loop has its own '
                            'measurement-noise stream (rng 42 seeded per '
                            'batch); cache rows are the full-run stream '
                            '(484 scenarios)'}
                print(f'[{vtag}] cache drift (subset):',
                      koop_drift[vtag]['pk_win_max_absdiff_fresh_vs_cache'],
                      flush=True)
        flags_k, met_k = row_metrics(pk_c, pk_c_run, dpcols)
        met_k['latency'] = dict(
            warm_median_ms=float(np.nanmedian(warm)),
            warm_mean_ms=float(np.nanmean(warm)),
            warm_p90_ms=float(np.nanpercentile(warm, 90)),
            warm_max_ms=float(np.nanmax(warm)),
            cold_median_ms=float(np.nanmedian(cold)),
            cold_mean_ms=float(np.nanmean(cold)),
            n_decisions_per_scenario=int(n_dec),
            n_warm_samples=int(warm.size),
            n_cold_samples=int(np.isfinite(cold).sum()),
            cold_sampling=f'every {COLD_SC}th scenario x every '
                          f'{COLD_DEC}th decision',
            note='fresh per-decision measurement: warm = bounds update + '
                 'solve on the scenario-warm-started solver; cold = fresh '
                 'OSQP setup + solve of the same problem')
        row_entry = dict(kind='sweep', latency=met_k.pop('latency'),
                         fit={'fit_json': lift.src_json,
                              'n_phi': int(lift.A.shape[0])})
        print(f'[{vtag}] done {time.time() - t_v:.0f}s — rescue '
              f'{int(flags_k["won_t"][need].sum())}/{int(need.sum())} '
              f'({met_k["rescue_true"]["value"]:.4f}), harmed '
              f'{met_k["harmed_pct"]["value"]:.2f}%, warm med '
              f'{row_entry["latency"]["warm_median_ms"]:.2f} ms', flush=True)
        row_entry['metrics'] = met_k
        rows_out[vtag] = row_entry
        json.dump({'vtag': vtag, 'metrics': met_k,
                   'need_sum': int(need.sum()), 'n_instants': n},
                  open(os.path.join(out_dir, f'_sweep_partial_{vtag}.json'),
                       'w'), indent=1, default=float)

    # ── scheduled-scan row (merged from the shipped one-attempt record) ─────
    scan_art_path = None
    for cand in (os.path.join(out_dir, 'task6_scan.json'),
                 os.path.join(DEFAULT_OUT, 'task6_scan.json')):
        if os.path.exists(cand):
            scan_art_path = cand
            break
    if 'scheduled_scan' in sel or (args.row == 'all' and not subset
                                   and scan_art_path):
        S_scan = json.load(open(scan_art_path)) if scan_art_path else {}
        if 'metrics' in S_scan:
            rows_out['scheduled_scan'] = dict(
                kind='sweep-stretch (ONE attempt, fixed recipe, no iteration)',
                metrics=S_scan['metrics'], recipe=S_scan['recipe'],
                identification=S_scan['identification'],
                n_candidates=S_scan['n_candidates'],
                n_identified_cleared=S_scan['n_identified_cleared'],
                fit='head ckpt ' + S_scan['ckpt'] + ' encoder (hypothesis, '
                    'alpha) per decision; min-norm Fibonacci direction/scale '
                    'scan under the IDENTIFIED faulted Yred at dt 5 ms, '
                    'held-to-clear')
        else:
            print('sched_scan row: task6_scan.json has no metrics (scan not '
                  'run) — row omitted')

    # ── assemble ─────────────────────────────────────────────────────────────
    table = {
        'meta': {
            'system': '9', 'split': args.split, 'seed_instants': 42,
            'instants_file': INSTANTS_DEFAULT,
            'instants_sha256': sha256_of(INSTANTS_DEFAULT),
            'n_instants_shared': int(len(KS_all)), 'n_true_instants': n,
            'subset': bool(subset),
            'n_scenarios_true_subset': int(len(np.unique(KS))),
            'dt_true': DT, 'grace_s': GRACE, 'omega_limit': LIMIT,
            'dp_hold': args.dp_hold, 'head_horizon': float(A['horizon']),
            'scorer': 'bus_9/koopman_baseline.py true-dynamics core (f64 '
                      'CPU, batched), cross-checked vs verify_true.py '
                      '(Task 5)',
            'crosscheck_json':
                os.path.join(out_dir, 'task5_crosscheck.json'),
            'bootstrap': {'B': BOOT, 'clusters': 'scenarios',
                          'seed': BOOT_SEED,
                          'note': 'compare_models protocol (rng 49, 2000 '
                                  'resamples, W over the full shared-draw '
                                  'unique scenario set)'},
            'head_columns_assembled': {
                'compare_out_s42/compare_results.json':
                    sha256_of(os.path.join(COMP, 'compare_results.json')),
                'shared_LSTMref_s42/verify_results.json':
                    sha256_of(os.path.join(BUS9,
                            'shared_LSTMref_s42/verify_results.json')),
                'shared_MLPref_s42/verify_results.json':
                    sha256_of(os.path.join(BUS9,
                            'shared_MLPref_s42/verify_results.json'))},
            'koopman_config': {
                'fits': {'koopman_gov_fullchan': 'task2_edmd_fullchan.json',
                         'korda_native_bare': 'task34_bare_refit.json'},
                'qp_knobs': dict(w=QP_W, db=QP_DB, Wt=QP_WT, qu=QP_QU,
                                 umax=QP_UMAX, rate_per_decision=QP_RATE,
                                 H_steps=H, decision_ms=80.0),
                'kf_q': KF_Q,
                'provenance': 'identical shipped Task-4 configuration '
                              '(task34_mpc.json); decisions every 20 ms '
                              'from t=0 over the whole window (Task-4 '
                              'protocol), application masked to '
                              "dp_hold='clear' (zeroed from the clear "
                              'instant); one measurement-noise stream '
                              '(seed 42) per batched closed-loop run',
                'latency_protocol': 'fresh per-decision (warm: bounds '
                                    'update + solve; cold: fresh setup + '
                                    f'solve, sampled every {COLD_SC}th '
                                    f'scenario x {COLD_DEC}th decision)',
            },
        },
        'rows': rows_out,
        'counts_reconcile': {
            'n_instants': n,
            'need_sum': int(need.sum()),
            'need_by_type': {tn: int(need[y_type == ti + 1].sum())
                             for ti, tn in enumerate(TYPE_NAMES)},
            'need_status3': int(need[DPS == 3].sum()),
            'excluded_no_active': fl_floor['excluded'],
            'note': 'floor-need flags are row-independent (shared draw, '
                    'same define) and equal for every my-row'},
        'gates': {'task5_crosscheck':
                  os.path.join(out_dir, 'task5_crosscheck.json')},
    }

    def metrics_of(row):
        return rows_out[row]['metrics'] if row in rows_out else None

    resc_or = metrics_of('oracle_label')['rescue_true']['value'] \
        if 'oracle_label' in rows_out else None
    resc_hd = rows_out['head_lstm_cascade']['metrics']['rescue_true']['value']
    gates = {}
    if resc_or is not None:
        ldc = label_design_check(S64, d, md, np.unique(KS))
        ldc['status1_design_rescue_within_margin'] = ldc[
            'design_peak_le_limit_minus_margin_frac']['status1']
        gates['oracle'] = {
            'oracle_rescue_true': resc_or,
            'floor_rescue_true_window': 0.0,
            'beats_floor_on_needy': bool(resc_or > 0.0),
            'head_rescue_true_artifact': resc_hd,
            'head_minus_oracle': resc_hd - resc_or,
            'age_structure': 'oracle rescue decays with fault age (label is '
                             'a SCHEDULED tk-blind correction; the head is '
                             'tk-adaptive) — see row age bins',
            'label_design_check': ldc,
            'consistent_with_wiki': bool(
                resc_or > 0.0
                and ldc.get('status1_design_rescue_le_limit_frac') is not None
                and ldc['status1_design_rescue_le_limit_frac'] >= 0.99
                and ldc['max_absdiff_vs_best_maxdev_window'] < 5e-4),
        }
        table['gates']['oracle'] = gates['oracle']
    if subset:
        table['gates']['subset_verification'] = {
            'n': n,
            'floor_oracle_fresh_cache_drift': {
                'floor': floor_drift, 'oracle': oracle_drift},
            'koopman_fresh_vs_cache': koop_drift,
            'note': 'floor/oracle: the fresh subset scorer is rerun on '
                    'those instants and compared to the shipped per-instant '
                    'caches; koopman: fresh subset closed-loop vs the '
                    'shipped full-run trajectory cache (measurement-noise '
                    'stream differs between batches — drift expected small '
                    'and recorded, not gated)'}
    suffix = f'_n{n}' if subset else ''
    path = os.path.join(out_dir, f'baseline_table{suffix}.json')
    json.dump(table, open(path, 'w'), indent=1, default=float)
    if gates:
        print(json.dumps(gates.get('oracle', {}),
                         indent=1, default=float))
    figure_quality(table, os.path.join(out_dir,
                                       f'task6_quality_bars{suffix}.png'))
    figure_latency(table, os.path.join(out_dir,
                                       f'task6_latency_table{suffix}.png'))
    print(f'TOTAL {time.time() - t0:.0f} s -> {path}')
    append_log(out_dir, ' '.join(sys.argv)) if not subset else None


# ══════════════════════════════════════════════════════════════════════════════
# Stretch — scheduled-scan row (ONE attempt, fixed recipe, no iteration);
# full seed-42 draw, writes task6_scan.json (sweep --row sched_scan merges it)
# ══════════════════════════════════════════════════════════════════════════════

def scan_candidates():
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                     [1, 1, 1], [-1, -1, -1], [1, -1, 0], [1, 0, -1]],
                    dtype=np.float64)
    dirs[6] /= np.sqrt(3); dirs[7] /= np.sqrt(3)
    dirs[8] /= np.sqrt(2); dirs[9] /= np.sqrt(2)
    scales = [SCAN_SCALE_MAX * f / FIB[-1] for f in FIB]
    return np.array([[s * v for v in dirs] for s in scales]).reshape(
        -1, 3), scales


@torch.no_grad()
def cmd_scan(args, out_dir):
    """STRETCH, single attempt: per-instant (hypothesis, alpha) from the head
    encoder, Fibonacci-scale/direction scan for the min-norm constant dP that
    clears the limit under the IDENTIFIED faulted Yred (dt 5 ms, judged window
    per valid_t), held-to-clear; winner applied + scored in TRUE dynamics.
    Fixed recipe — no iteration."""
    t0 = time.time()
    try:
        ckpt = os.path.join(BUS9, args.ckpt)
        KS_all, TK_all, inst = load_instants(INSTANTS_DEFAULT)
        md = load_meta(); d = load_data()
        A, _ck = ckpt_args(ckpt)
        LIMIT = float(md['omega_limit'])
        DPS = np.asarray(md['dp_status'])[KS_all[:N_TRUE]]
        KS, TK = KS_all[:N_TRUE], TK_all[:N_TRUE]
        active, vt = build_windows(A, d, KS, TK)
        boot_weights(KS_all)
        _BOOT['inv_used'] = _BOOT['inv'][:N_TRUE]
        y_age = TK - d['t_fault'][KS]
        rows_meta = (d['y_type'][KS], y_age, DPS)
        S64 = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
                    os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
                    dtype=torch.float64, lines=d['lines'])
        vtm = import_verify_true(ckpt, out_dir)
        rng11 = np.random.default_rng(11)

        # ── identified (hyp, alpha) per instant (encoder forward, chunked) ──
        hyp_id = np.empty(N_TRUE, dtype=int); det_arr = np.zeros(N_TRUE, bool)
        al_hat = np.empty(N_TRUE)
        for s in range(0, N_TRUE, 256):
            sl = slice(s, min(s + 256, N_TRUE))
            b = vtm.make_batch(KS[sl], rng11, tk_fixed=TK[sl])
            o = vtm.model(b[0].to(vtm.device), b[1].to(vtm.device),
                          hard=True, ctrl=True,
                          vmask=b[3].to(vtm.device))
            lg = o['hyp_logits']
            det_t = lg[:, 1:].logsumexp(1) > lg[:, 0]
            pf = lg[:, 1:].argmax(1) + 1
            pred_hard = torch.where(det_t, pf, torch.zeros_like(pf))
            det_arr[sl] = det_t.cpu().numpy().astype(bool)
            hyp_id[sl] = pred_hard.cpu().numpy() - 1      # -1 => class 0
            al_hat[sl] = o['alpha'].cpu().numpy()
            print(f'  fwd chunk {sl.stop}/{N_TRUE}', flush=True)
        print(f'detected: {det_arr.sum()}/{N_TRUE}; '
              f'right-type among detected: '
              f"{(hyp_id[det_arr] == d['y_type'][KS[det_arr]] - 1).mean():.4f}")

        # identified faulted networks per instant (build once)
        ksid = KS.copy()
        h_idx = np.clip(hyp_id, 0, len(HYPOTHESES) - 1)
        Yf_id = torch.zeros(N_TRUE, 3, 3, dtype=S64.cdtype)
        dPm_id = torch.zeros(N_TRUE, 3, dtype=S64.rdtype)
        for j, k in enumerate(ksid):
            ha = torch.tensor([float(al_hat[j])], dtype=S64.rdtype)
            if det_arr[j]:
                Yf_id[j], _ = S64.kron_reduce(S64.faulted_Y(int(h_idx[j]),
                                                            ha))
                dPm_id[j] = S64.dPm0(int(h_idx[j]), ha)[0]
        print('identified networks built', flush=True)

        cands, scales = scan_candidates()
        n_c = len(cands)
        print(f'scan candidates: {n_c} (dirs 10 x Fibonacci scales '
              f'{len(scales)})')
        winner = np.zeros((N_TRUE, 3))
        winner_from = np.full(N_TRUE, -1, dtype=int)
        pk_id_win = np.full(N_TRUE, -1.0)
        for s in range(0, N_TRUE, SCAN_CHUNK):
            sl = slice(s, min(s + SCAN_CHUNK, N_TRUE))
            n_i = sl.stop - sl.start
            ks_c = KS[sl]; tk_c = TK[sl]; vt_c = vt[sl]
            n_r = n_i * n_c
            # replicated rows: instant-major, candidate-minor
            ks_rep = np.repeat(ks_c, n_c)
            tk_rep = np.repeat(tk_c, n_c)
            vt_rep = np.repeat(vt_c, n_c)
            dP_rep = np.tile(cands, (n_i, 1))
            Yf_rep = torch.repeat_interleave(Yf_id[sl], n_c, dim=0)
            dPm_rep = torch.repeat_interleave(dPm_id[sl], n_c, dim=0)
            # full run under IDENTIFIED networks, dP applied [tk, tc)
            n_steps = 1600
            tf_rep = torch.repeat_interleave(
                torch.as_tensor(np.asarray(d['t_fault'])[ks_c],
                                dtype=S64.rdtype), n_c)
            tc_rep = torch.repeat_interleave(
                torch.as_tensor(np.asarray(d['t_clear'])[ks_c],
                                dtype=S64.rdtype), n_c)
            tk_t = torch.as_tensor(tk_rep, dtype=S64.rdtype)
            tc_off = torch.repeat_interleave(
                torch.as_tensor(np.asarray(d['t_clear'])[ks_c],
                                dtype=S64.rdtype), n_c)
            om_t, _ = full_sim_np(S64, tf_rep, tc_rep, tk_t, tc_off, Yf_rep,
                                  dPm_rep,
                                  torch.as_tensor(dP_rep, dtype=S64.rdtype),
                                  DT, n_steps, want_states=False)
            pk_id, _ = peaks_from_om(om_t, tk_rep, vt_rep, DT,
                                     device=S64.device)
            pk = pk_id.cpu().numpy().reshape(n_i, n_c)
            norm2 = np.linalg.norm(dP_rep, axis=1).reshape(n_i, n_c)
            for j in range(n_i):
                if not det_arr[sl][j]:
                    continue
                okrow = pk[j] <= LIMIT
                if okrow.any():
                    jj = np.where(okrow)[0][np.argmin(norm2[j][okrow])]
                    pk_id_win[sl][j] = pk[j][jj]
                else:
                    jj = int(np.argmin(pk[j] * 1e4 + norm2[j]))
                    pk_id_win[sl][j] = pk[j][jj]
                winner_from[sl][j] = jj
                winner[sl][j] = dP_rep[j * n_c + jj]
            print(f'  scan chunk (identified) {sl.stop}/{N_TRUE}', flush=True)
        n_ident_cleared = int((pk_id_win <= LIMIT).sum())

        pk_tr = np.empty(N_TRUE); pk_tr_run = np.empty(N_TRUE)
        pk_free = np.empty(N_TRUE); pk_free_run = np.empty(N_TRUE)
        x_at_all = np.empty((N_TRUE, 9))
        for s in range(0, N_TRUE, CHUNK):
            sl = slice(s, min(s + CHUNK, N_TRUE))
            pw, pr, xa, _, _ = run_true_batched(S64, d, KS[sl], TK[sl],
                                                vt[sl])
            pk_free[sl] = pw; pk_free_run[sl] = pr; x_at_all[sl] = xa
            pwt, prt, _, _, _ = run_true_batched(S64, d, KS[sl], TK[sl],
                                                 vt[sl], dP=winner[sl])
            pk_tr[sl] = pwt; pk_tr_run[sl] = prt
            print(f'  scan chunk (true+free) {sl.stop}/{N_TRUE}', flush=True)
        fl = case_flags(pk_free, pk_free_run, x_at_all, TK, active, LIMIT)
        flags_sc = dict(need=fl['need'], healthy=fl['healthy'],
                        reach=fl['reach'], won_t=pk_tr <= LIMIT,
                        run_ok=pk_tr_run <= LIMIT,
                        free_run_ok=(pk_free_run <= LIMIT),
                        harmed=fl['healthy'] & (pk_tr > LIMIT))
        dp_sc = {'l2_first': np.linalg.norm(winner, axis=1),
                 'l2_max': np.linalg.norm(winner, axis=1),
                 'l2_mean': np.linalg.norm(winner, axis=1),
                 'l1_mean': np.abs(winner).sum(1)}
        met_sc = rescue_metrics(flags_sc, rows_meta, dp_sc)
        art = {
            'cmd': ' '.join(sys.argv), 'ckpt': ckpt,
            'recipe': dict(dirs=SCAN_DIR_NAMES,
                           scales=[round(x, 6) for x in scales],
                           selection='min ||dP|| among identified-sim '
                                     'clearers (peak <= limit in the judged '
                                     'window under the IDENTIFIED Yred); '
                                     'else argmin peak then norm',
                           one_attempt=True),
            'n_candidates': n_c,
            'n_identified_cleared': n_ident_cleared,
            'identification': {
                'detected': int(det_arr.sum()),
                'type_acc_among_detected': float(
                    (hyp_id[det_arr] == d['y_type'][KS[det_arr]] - 1).mean())},
            'metrics': met_sc,
            'runtime_s': time.time() - t0,
        }
        json.dump(art, open(os.path.join(out_dir, 'task6_scan.json'), 'w'),
                  indent=1, default=float)
        print('SCAN RESULT: rescue_true '
              f"{met_sc['rescue_true']['value']:.4f} (n_need "
              f"{int(flags_sc['need'].sum())}); saved task6_scan.json")
        print(f'TOTAL {time.time() - t0:.0f} s')
    except Exception as e:                       # one attempt only: log + skip
        import traceback
        traceback.print_exc()
        json.dump({'scan': 'SKIPPED — single attempt failed',
                   'error': repr(e),
                   'traceback': traceback.format_exc()},
                  open(os.path.join(out_dir, 'task6_scan.json'), 'w'),
                  indent=1)
        print('scan SKIPPED (logged); sweep table unaffected')


# ══════════════════════════════════════════════════════════════════════════════
# argparse — consolidated flag system
# ══════════════════════════════════════════════════════════════════════════════

def add_shared(p, n_default=None):
    p.add_argument('--instants-file', default=INSTANTS_DEFAULT,
                   help='shared-draw instants JSON (seed 42, split test)')
    p.add_argument('--split', default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='_koopman_out',
                   help='output/scratch dir relative to bus_9 (default '
                        'bus_9/_koopman_out; shipped artifacts live here)')
    p.add_argument('--dp-hold', choices=['clear'], default='clear',
                   help='ΔP application semantics (shipped: held to the '
                        'clear instant; other releases not defined)')
    if n_default is not None:
        p.add_argument('--n', type=int, default=n_default)
    return p


def main():
    # Windows consoles/pipes default to cp1252; this script prints ω/Δ glyphs —
    # force UTF-8 so --help and run output render anywhere.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    ap = argparse.ArgumentParser(
        prog='koopman_baseline.py',
        description='9-bus Koopman-MPC baseline (Korda/Susuki/Mezic 2018, '
                    'arXiv:1803.10744) — consolidated committed entry '
                    'point. Shipped configs frozen in bus_9/_koopman_out. '
                    'Wiki: PINN wiki/queries/koopman-mpc-baseline-plan.md')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('smoke', help='Task 1: clean-dynamics sampler check')
    add_shared(p)
    p.add_argument('--n_per_type', type=int, default=2)
    p.set_defaults(fn=lambda a, o: cmd_smoke(a, o))

    p = sub.add_parser('fit-edmd',
                       help='Task 2: EDMD lifted linear predictor fits')
    add_shared(p)
    p.add_argument('--fit', required=True,
                   choices=['fullchan', 'angdiff2', 'bare'])
    p.add_argument('--n_train', type=int, default=216000)
    p.add_argument('--n_val', type=int, default=24000)
    p.add_argument('--n_episodes', type=int, default=4000)
    p.add_argument('--ep_len', type=int, default=200)
    p.add_argument('--roll_steps', type=int, default=200)
    p.add_argument('--n_roll_states', type=int, default=300)
    p.add_argument('--dt', type=float, default=DT_TRUE)
    p.set_defaults(fn=lambda a, o: cmd_fit_edmd(a, o))

    p = sub.add_parser('fit-kf', help='Task 3: steady lifted Kalman filter')
    add_shared(p)
    p.set_defaults(fn=lambda a, o: cmd_fit_kf(a, o))

    p = sub.add_parser(
        'run-mpc', help='Task 4: dense-form QP MPC closed loop (osqp)')
    add_shared(p)
    p.add_argument('--channel',
                   choices=['gov_fullchan', 'korda_native_bare', 'both'],
                   default='both',
                   help='control channel variant (shipped default: both)')
    p.set_defaults(fn=lambda a, o: cmd_run_mpc(a, o))

    p = sub.add_parser('sweep', help='Task 6: seed-42 validation sweep table')
    add_shared(p)
    p.add_argument('--row',
                   choices=['all', 'koopman_gov', 'korda_native', 'oracle',
                            'floor', 'sched_scan'], default='all')
    p.add_argument('--n', type=int, default=None,
                   help='subset of the shared draw (e.g. --n 16 for the '
                        'verification subset; default: full 2048-instant '
                        'true subset)')
    p.add_argument('--ckpt', default='receding_cascade/refined/best.pt')
    p.add_argument('--no-cache', dest='cache', action='store_false',
                   help='ignore the shipped trajectory caches and recompute')
    p.set_defaults(fn=lambda a, o: cmd_sweep(a, o), cache=True)

    p = sub.add_parser('scan', help='Stretch: scheduled-scan row (full draw)')
    add_shared(p)
    p.add_argument('--ckpt', default='receding_cascade/refined/best.pt')
    p.set_defaults(fn=lambda a, o: cmd_scan(a, o))

    p = sub.add_parser('crosscheck',
                       help='Task 5: scorer vs verify_true.py (GATE)')
    add_shared(p, n_default=16)
    p.add_argument('--ckpt', default='receding_cascade/refined/best.pt')
    p.set_defaults(fn=lambda a, o: cmd_crosscheck(a, o))

    args = ap.parse_args()
    if args.instants_file != INSTANTS_DEFAULT:
        # alternate instants file: accepted but gated (shipped protocol pins
        # the seed-42 shared draw)
        raise SystemExit('shipped protocol pins the seed-42 shared draw '
                         '(compare_out_s42/instants.json); alternate '
                         'instants files are not defined for this baseline')
    out_dir = resolve_out(args)
    args.fn(args, out_dir)


if __name__ == '__main__':
    main()
