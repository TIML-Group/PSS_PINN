"""
pinn_multifault.py
==================
Trains three separate PINNs — one per fault type:
    0 — Load faults   → pinn_output_load/best_model.pt
    1 — Gen trips     → pinn_output_gen/best_model.pt
    2 — Line faults   → pinn_output_line/best_model.pt

Each model has fault-type-specific inputs:
    Load PINN:  [t, alpha, fault_loc, t_clear]  — 4 inputs
    Gen PINN:   [t, alpha, fault_loc, t_clear]  — 4 inputs
    Line PINN:  [t, fault_loc, t_clear]         — 3 inputs (alpha always 1.0, dropped)

All models share:
    - 14 outputs: [delta_2_rel, delta_3_rel, omega_0..2, Pm_0..2, Id_0..2, Iq_0..2]
    - Same physics loss (delta, Pm with time-varying Pm0, omega ODEs)
    - Same stratified train/val split (90% train, 10% val)
    - Same architecture: hidden layers of 128 neurons

Usage:
    python pinn_multifault.py --data training_data_v4 --epochs 10000
    python pinn_multifault.py --data training_data_v4 --epochs 10000 --fault_type 1
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from networks import DNN

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',            type=str,   default='training_data_v4')
parser.add_argument('--epochs',          type=int,   default=10000)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--batch_size',      type=int,   default=1024)
parser.add_argument('--hidden',          type=int,   default=128)
parser.add_argument('--n_layers',        type=int,   default=5)
parser.add_argument('--val_frac',        type=float, default=0.1)
parser.add_argument('--w_physics',       type=float, default=0.001)
parser.add_argument('--w_physics_start', type=int,   default=2000)
parser.add_argument('--n_colloc',        type=int,   default=1024)
parser.add_argument('--save_every',      type=int,   default=500)
parser.add_argument('--fault_type',      type=int,   default=-1,
                    help='Train only this fault type (0=load,1=gen,2=line). Default: all three.')
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── Physical constants ─────────────────────────────────────────────────────────
H        = torch.tensor([23.64, 6.40, 3.01], dtype=torch.float32, device=device)
D        = torch.tensor([2.00,  2.00, 2.00],  dtype=torch.float32, device=device)
f0       = 60.0
wb       = 2 * np.pi * f0
R        = 0.05
Tg       = 0.2
t_fault  = 0.2

# ── Load full dataset ──────────────────────────────────────────────────────────
X_full = np.load(os.path.join(args.data, 'X_train.npy')).astype(np.float32)
Y_full = np.load(os.path.join(args.data, 'Y_train.npy')).astype(np.float32)
Y_full = Y_full[:, 1:]   # drop delta_1_rel (always 0)
labels_full = np.load(os.path.join(args.data, 'stable_labels.npy'))

with open(os.path.join(args.data, 'meta.json')) as f:
    meta = json.load(f)

Eq_prime_np = np.array(meta['Eq_prime'], dtype=np.float32)
Ed_prime_np = np.array(meta['Ed_prime'], dtype=np.float32)

# Pm0 from power flow (actual governor setpoint)
t0_mask  = np.abs(X_full[:, 0]) < 1e-6
Pm0_vals = Y_full[t0_mask, 5:8].mean(axis=0).tolist()
print(f"Pm0 (governor setpoint): {Pm0_vals}")

# Y column layout (14 cols after dropping delta_1_rel):
# 0,1    -> delta_2_rel, delta_3_rel
# 2,3,4  -> omega_0, omega_1, omega_2
# 5,6,7  -> Pm_0, Pm_1, Pm_2
# 8,9,10 -> Id_0, Id_1, Id_2
# 11,12,13 -> Iq_0, Iq_1, Iq_2
Y_cols = meta['Y_train_cols'][1:]

# ── Fault type config ──────────────────────────────────────────────────────────
fault_configs = {
    0: {'name': 'load_fault', 'outdir': 'pinn_output_load',
        'input_cols': [0, 1, 2, 4],        # t, alpha, fault_loc, t_clear
        'col_names':  ['t', 'alpha', 'fault_loc', 't_clear'],
        'n_inputs': 4},
    1: {'name': 'gen_trip',   'outdir': 'pinn_output_gen',
        'input_cols': [0, 1, 2, 4],        # t, alpha, fault_loc, t_clear
        'col_names':  ['t', 'alpha', 'fault_loc', 't_clear'],
        'n_inputs': 4},
    2: {'name': 'transmission_line_fault', 'outdir': 'pinn_output_line',
        'input_cols': [0, 2, 4],           # t, fault_loc, t_clear (drop alpha=always 1.0)
        'col_names':  ['t', 'fault_loc', 't_clear'],
        'n_inputs': 3},
    3: {'name': 'transformer_fault', 'outdir': 'pinn_output_transformer',
        'input_cols': [0, 2, 4],           # t, fault_loc, t_clear (drop alpha=always 1.0)
        'col_names':  ['t', 'fault_loc', 't_clear'],
        'n_inputs': 3},
}

fault_types_to_train = [args.fault_type] if args.fault_type >= 0 else [0, 1, 2, 3]

# ── Scenario boundaries ────────────────────────────────────────────────────────
t_col  = X_full[:, 0]
resets = np.where(np.diff(t_col) < 0)[0] + 1
resets = np.concatenate([[0], resets, [len(t_col)]])
n_sc   = len(resets) - 1

sc_fault  = np.array([int(X_full[resets[sc], 3]) for sc in range(n_sc)])
sc_stable = np.array([int(labels_full[resets[sc]]) for sc in range(n_sc)])


# Train one model per fault type

for ft in fault_types_to_train:
    cfg = fault_configs[ft]
    print(f"\n{'='*70}")
    print(f"  TRAINING: {cfg['name'].upper()}  (fault_type_id = {ft})")
    print(f"  Inputs:   {cfg['col_names']}")
    print(f"  Output:   {args.outdir if hasattr(args,'outdir') else cfg['outdir']}")
    print(f"{'='*70}\n")

    os.makedirs(cfg['outdir'], exist_ok=True)

    # ── Filter data for this fault type ───────────────────────────────────────
    ft_sc_mask = sc_fault == ft

    # For line faults: exclude locations that are always (or almost always)
    # unstable regardless of t_clear -- they provide no stable reference and
    # confuse the model.
    #   67  = line Bus6-Bus7 (1/38 stable in the expanded v2 line dataset)
    #   16  = Gen1's step-up transformer (0/38 stable -- disconnecting a
    #         generator's only grid connection is far more severe than a
    #         transmission line loss; confirmed via generate_data_line_only.py's
    #         v2 run, worst deviation 8.2% vs <1.5% for transmission lines)
    #   28  = Gen2's step-up transformer (0/38 stable, same reasoning)
    # Gen0's transformer (loc 3) is NOT excluded -- it has 2/38 stable
    # scenarios, thin but nonzero, so it stays in training (though note:
    # after excluding 16/28, fault_type 3 has only this ONE usable location
    # left -- a physical limit of the system, not fixable with more data).
    #
    # ft==2 and ft==3 are now SEPARATE categories (see
    # split_line_and_transformer.py) -- transmission lines and transformers
    # used to be trained together as one "line_fault" type, which mixed a
    # mild severity regime with one 5-10x more severe, visibly breaking
    # training (val_loss got WORSE with more data: 0.257 -> 0.295, with
    # wild non-converging validation swings).
    # ft=3 (transformers): previously excluded 16.0/28.0 (Gen1/Gen2's
    # transformers, always unstable), leaving only Gen0's -- but that made
    # fault_loc a CONSTANT (always 3.0) across the whole training set, so
    # the network had zero signal to learn location-dependence from at all.
    # Re-including them gives fault_loc genuine variation (3, 16, 28), which
    # may help the model learn the shape of that dependence even though 2 of
    # the 3 locations never produce a stable outcome.
    EXCLUDED_LOCS = {2: [67.0]}
    if ft in EXCLUDED_LOCS:
        sc_loc = np.array([X_full[resets[sc], 2] for sc in range(n_sc)])
        ft_sc_mask = ft_sc_mask & ~np.isin(sc_loc, EXCLUDED_LOCS[ft])
        print(f"  Note: excluding locations {EXCLUDED_LOCS[ft]} "
              f"(always/near-always unstable — no stable scenarios possible)")

    ft_sc_idx  = np.where(ft_sc_mask)[0]

    # Get all rows for this fault type
    ft_rows = []
    for sc in ft_sc_idx:
        ft_rows.extend(range(resets[sc], resets[sc+1]))
    ft_rows = np.array(ft_rows)

    X_ft = X_full[ft_rows][:, cfg['input_cols']]  # select relevant input cols
    Y_ft = Y_full[ft_rows]
    L_ft = labels_full[ft_rows]

    n_ft_sc = len(ft_sc_idx)
    print(f"  Scenarios: {n_ft_sc}  Rows: {len(ft_rows)}")

    # ── Normalisation (per fault type) ────────────────────────────────────────
    X_min_ft   = X_ft.min(axis=0)
    X_max_ft   = X_ft.max(axis=0)
    X_range_ft = np.where(X_max_ft - X_min_ft > 1e-8, X_max_ft - X_min_ft, 1.0)
    X_norm_ft  = (X_ft - X_min_ft) / X_range_ft

    Y_mean_ft = Y_ft.mean(axis=0)
    Y_std_ft  = np.where(Y_ft.std(axis=0) > 1e-8, Y_ft.std(axis=0), 1.0)
    Y_norm_ft = (Y_ft - Y_mean_ft) / Y_std_ft

    Y_mean_t = torch.tensor(Y_mean_ft, dtype=torch.float32, device=device)
    Y_std_t  = torch.tensor(Y_std_ft,  dtype=torch.float32, device=device)

    print(f"  Input normalisation:")
    for i, col in enumerate(cfg['col_names']):
        print(f"    {col}: [{X_min_ft[i]:.4f}, {X_max_ft[i]:.4f}]")

    # ── Stratified train/val split (within this fault type) ──────────────────
    sc_stable_ft = sc_stable[ft_sc_mask]
    strat_key_ft = sc_stable_ft  

    np.random.seed(42)
    val_sc_ft   = []
    train_sc_ft = []
    for key in np.unique(strat_key_ft):
        group = np.where(strat_key_ft == key)[0]
        np.random.shuffle(group)
        n_val = max(1, int(len(group) * args.val_frac))
        val_sc_ft.extend(group[:n_val].tolist())
        train_sc_ft.extend(group[n_val:].tolist())

    # Convert local sc indices back to row indices
    def ft_sc_to_rows(local_sc_indices):
        rows = []
        for lsc in local_sc_indices:
            sc = ft_sc_idx[lsc]
            rows.extend(range(resets[sc], resets[sc+1]))
        return np.array(rows)

    train_rows_ft = ft_sc_to_rows(train_sc_ft)
    val_rows_ft   = ft_sc_to_rows(val_sc_ft)

    # Map ft rows back to indices in X_ft/Y_ft arrays
    ft_rows_set   = {r: i for i, r in enumerate(ft_rows)}
    train_idx_ft  = np.array([ft_rows_set[r] for r in train_rows_ft])
    val_idx_ft    = np.array([ft_rows_set[r] for r in val_rows_ft])

    X_tr  = torch.tensor(X_norm_ft[train_idx_ft], dtype=torch.float32)
    Y_tr  = torch.tensor(Y_norm_ft[train_idx_ft], dtype=torch.float32)
    X_val = torch.tensor(X_norm_ft[val_idx_ft],   dtype=torch.float32)
    Y_val = torch.tensor(Y_norm_ft[val_idx_ft],   dtype=torch.float32)

    print(f"  Train: {len(train_sc_ft)} scenarios ({len(train_idx_ft)} rows)")
    print(f"  Val:   {len(val_sc_ft)} scenarios ({len(val_idx_ft)} rows)")
    print(f"    stable in val: {sc_stable_ft[val_sc_ft].sum()}  "
          f"unstable: {(sc_stable_ft[val_sc_ft]==0).sum()}")

    # ── Model ────────────────────────────────────────────────────────────────
    layers = [cfg['n_inputs']] + [args.hidden] * args.n_layers + [14]
    model  = DNN(layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {layers}  |  Parameters: {n_params:,}")

    # ── Physics loss collocation data ─────────────────────────────────────────
    # Keep raw X for physics loss (need physical units for Pm0 time-varying logic)
    X_train_phys_ft = X_ft[train_idx_ft]  # physical units, fault-type input cols

    Eq_prime = torch.tensor(Eq_prime_np, dtype=torch.float32, device=device)
    Ed_prime = torch.tensor(Ed_prime_np, dtype=torch.float32, device=device)
    Pm0      = torch.tensor(Pm0_vals,    dtype=torch.float32, device=device)

    def physics_loss(model, n_colloc):
        idx   = np.random.randint(0, len(X_train_phys_ft), n_colloc)
        x_raw = X_train_phys_ft[idx].copy()

        # Randomise t over full domain (t is always first input col)
        t_max = float(X_max_ft[0])
        x_raw[:, 0] = np.random.uniform(0.0, t_max, n_colloc)

        x_norm = (x_raw - X_min_ft) / X_range_ft
        x_t    = torch.tensor(x_norm, dtype=torch.float32, device=device,
                               requires_grad=True)

        y_norm = model(x_t)
        y_phys = y_norm * Y_std_t + Y_mean_t

        t_scale = float(X_range_ft[0])  # t range for denormalising derivative

        def ddt(y_col):
            g = torch.autograd.grad(y_col.sum(), x_t,
                                    create_graph=True, retain_graph=True)[0][:, 0]
            return g / t_scale

        omega_0 = y_phys[:, 2]

        # Delta ODE (normalized by wb)
        r_d2 = (ddt(y_phys[:, 0]) / wb - (y_phys[:, 3] - omega_0)) ** 2
        r_d3 = (ddt(y_phys[:, 1]) / wb - (y_phys[:, 4] - omega_0)) ** 2
        loss_delta = (r_d2 + r_d3).mean()

        # Time-varying Pm0 for gov ODE
        t_vals = torch.tensor(x_raw[:, 0], dtype=torch.float32, device=device)

        # t_clear and alpha col indices depend on fault type
        # For load/gen: input_cols = [t, alpha, fault_loc, t_clear] → alpha=col1, t_clear=col3
        # For line/transformer (ft 2, 3): input_cols = [t, fault_loc, t_clear] → no alpha, t_clear=col2
        if ft in (2, 3):  # transmission line or transformer fault -- same 3-input schema
            t_clear_vals = torch.tensor(x_raw[:, 2], dtype=torch.float32, device=device)
            alpha_vals   = torch.ones(n_colloc, device=device)   # alpha=1 always
        else:
            alpha_vals   = torch.tensor(x_raw[:, 1], dtype=torch.float32, device=device)
            t_clear_vals = torch.tensor(x_raw[:, 3], dtype=torch.float32, device=device)

        fault_active = (t_vals > t_fault) & (t_vals < t_clear_vals)
        is_gen_trip  = (ft == 1)  # boolean scalar

        # Fault loc col index
        loc_col = 1 if ft in (2, 3) else 2
        fault_loc = torch.tensor(x_raw[:, loc_col], dtype=torch.float32, device=device)

        res_Pm = []
        for i in range(3):
            Pm0_i = Pm0[i] * torch.ones(n_colloc, device=device)
            if is_gen_trip:
                gen_i_mask = (fault_loc == float(i)) & fault_active
                Pm0_i = torch.where(gen_i_mask,
                                    Pm0[i] * (1.0 - alpha_vals),
                                    Pm0_i)
            dPm_dt = ddt(y_phys[:, 5 + i])
            Pm_ref = Pm0_i - (y_phys[:, 2 + i] - 1.0) / R
            rhs    = (Pm_ref - y_phys[:, 5 + i]) / Tg
            res_Pm.append((dPm_dt - rhs) ** 2)
        loss_Pm = torch.stack(res_Pm).mean()

        res_omega = []
        for i in range(3):
            domega_dt = ddt(y_phys[:, 2 + i])
            Pe_i      = Ed_prime[i]*y_phys[:, 8+i] + Eq_prime[i]*y_phys[:, 11+i]
            rhs       = (y_phys[:, 5+i] - Pe_i - D[i]*(y_phys[:, 2+i]-1.0)) / (2.0*H[i])
            res_omega.append((domega_dt - rhs) ** 2)
        loss_omega = torch.stack(res_omega).mean()

        total = loss_delta + loss_Pm + loss_omega
        return total, loss_delta.item(), loss_Pm.item(), loss_omega.item()

    # ── Training setup ────────────────────────────────────────────────────────
    train_ds = TensorDataset(X_tr, Y_tr)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=500, factor=0.5, min_lr=1e-5)
    loss_fn   = nn.MSELoss()

    X_val_dev = X_val.to(device)
    Y_val_dev = Y_val.to(device)

    best_val     = float('inf')
    train_losses = []; val_losses = []; phys_losses = []
    delta_losses = []; Pm_losses  = []; omega_losses = []

    t_start = time.time()
    print(f"\n  Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        use_physics = (epoch >= args.w_physics_start) and (args.w_physics > 0)

        model.train()
        batch_losses = []; batch_phys = []
        batch_delta  = []; batch_Pm   = []; batch_omega = []

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred      = model(xb)
            data_loss = loss_fn(pred, yb)

            if use_physics:
                p_loss, l_d, l_pm, l_om = physics_loss(model, args.n_colloc)
                loss = data_loss + args.w_physics * p_loss
                batch_phys.append(p_loss.item())
                batch_delta.append(l_d); batch_Pm.append(l_pm); batch_omega.append(l_om)
            else:
                loss = data_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(data_loss.item())

        train_loss = float(np.mean(batch_losses))
        phys_mean  = float(np.mean(batch_phys))  if batch_phys  else 0.0
        d_mean     = float(np.mean(batch_delta))  if batch_delta else 0.0
        pm_mean    = float(np.mean(batch_Pm))     if batch_Pm    else 0.0
        om_mean    = float(np.mean(batch_omega))  if batch_omega else 0.0

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_dev), Y_val_dev).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss); val_losses.append(val_loss)
        phys_losses.append(phys_mean);  delta_losses.append(d_mean)
        Pm_losses.append(pm_mean);      omega_losses.append(om_mean)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'layers': layers, 'fault_type': ft,
                'fault_name': cfg['name'], 'input_cols': cfg['input_cols'],
                'col_names': cfg['col_names'],
                'X_min': X_min_ft, 'X_max': X_max_ft, 'X_range': X_range_ft,
                'Y_mean': Y_mean_ft, 'Y_std': Y_std_ft, 'meta': meta,
                'val_loss': val_loss, 'Pm0_vals': Pm0_vals,
                'train_losses': train_losses, 'val_losses': val_losses,
            }, os.path.join(cfg['outdir'], 'best_model.pt'))

        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            lr_now  = optimiser.param_groups[0]['lr']
            if use_physics:
                print(f"  Epoch {epoch:6d}  train={train_loss:.6f}  val={val_loss:.6f}"
                      f"  best={best_val:.6f}  phys={phys_mean:.4f}"
                      f"  [delta={d_mean:.7f}  Pm={pm_mean:.6f}  omega={om_mean:.8f}]"
                      f"  lr={lr_now:.2e}  ({elapsed:.0f}s)")
            else:
                print(f"  Epoch {epoch:6d}  train={train_loss:.6f}  val={val_loss:.6f}"
                      f"  best={best_val:.6f}  lr={lr_now:.2e}  ({elapsed:.0f}s)")

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'layers': layers, 'fault_type': ft,
                'X_min': X_min_ft, 'X_max': X_max_ft, 'X_range': X_range_ft,
                'Y_mean': Y_mean_ft, 'Y_std': Y_std_ft, 'meta': meta,
            }, os.path.join(cfg['outdir'], f'checkpoint_ep{epoch}.pt'))

    print(f"\n  Done. Best val loss: {best_val:.6f}")
    print(f"  Total time: {time.time()-t_start:.1f}s")

    # Per-output RMSE
    print(f"\n  ── Per-output val RMSE ({cfg['name']}) ──")
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_val_dev).cpu().numpy() * Y_std_ft + Y_mean_ft
    Y_true = Y_val.numpy() * Y_std_ft + Y_mean_ft
    for i, col in enumerate(Y_cols):
        rmse = np.sqrt(np.mean((Y_pred[:, i] - Y_true[:, i])**2))
        unit = 'rad' if 'delta' in col else 'pu'
        print(f"    {col:20s}  RMSE = {rmse:.6f} {unit}")

print(f"\n{'='*70}")
print("All models trained. Saved to:")
for ft in fault_types_to_train:
    print(f"  {fault_configs[ft]['outdir']}/best_model.pt")