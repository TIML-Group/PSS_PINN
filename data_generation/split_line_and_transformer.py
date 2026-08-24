"""
split_line_and_transformer.py
================================
Splits fault_type_id=2 ("line_fault") into two SEPARATE categories:
  - fault_type_id=2: transmission lines only (34, 35, 46, 58, 67, 78)
  - fault_type_id=3: generator transformers only (3, 16, 28)

WHY: transformer faults produce deviations 5-10x larger than transmission
line faults (confirmed: transmission lines topped out ~1.4%, transformers
reached 6.9-8.2%, Pm even went negative). Training them together as one
"line_fault" category means one shared Y-normalization and one shared loss
has to fit both a mild and a severe regime at once -- this visibly broke
training (val_loss got WORSE with the transformer data added: 0.257 -> 0.295,
with wild, non-converging validation loss swings). Separating them into two
independently-normalized categories should fix this.

NOTE: after this split, and after excluding the always-unstable locations
(67, 16, 28) in the training script, the transformer category (3) will only
have ONE usable location left (Gen0's transformer, loc=3) -- a physical
limit of this system (only 3 transformers exist, 2 are always unstable),
not something more data can fix. Set expectations accordingly.

Usage:
    python split_line_and_transformer.py --data_dir training_data_merged_v2 --out_dir training_data_split
"""

import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir',  type=str, default='training_data_split')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

X = np.load(os.path.join(args.data_dir, 'X_train.npy'))
Y = np.load(os.path.join(args.data_dir, 'Y_train.npy'))
stable = np.load(os.path.join(args.data_dir, 'stable_labels.npy'))
with open(os.path.join(args.data_dir, 'meta.json')) as f:
    meta = json.load(f)

FAULT_TYPE_COL = 3
LOC_COL = 2
LINE_TYPE_ID = 2.0
TRANSFORMER_LOCS = [3.0, 16.0, 28.0]

is_old_line_type = X[:, FAULT_TYPE_COL] == LINE_TYPE_ID
is_transformer_loc = np.isin(X[:, LOC_COL], TRANSFORMER_LOCS)

n_before = {
    'transmission_line_rows': int((is_old_line_type & ~is_transformer_loc).sum()),
    'transformer_rows': int((is_old_line_type & is_transformer_loc).sum()),
}
print(f"Rows being relabeled:")
print(f"  Transmission lines (staying fault_type_id=2): {n_before['transmission_line_rows']}")
print(f"  Transformers (relabeled to fault_type_id=3):  {n_before['transformer_rows']}")

X_new = X.copy()
X_new[is_old_line_type & is_transformer_loc, FAULT_TYPE_COL] = 3.0

np.save(os.path.join(args.out_dir, 'X_train.npy'), X_new)
np.save(os.path.join(args.out_dir, 'Y_train.npy'), Y)
np.save(os.path.join(args.out_dir, 'stable_labels.npy'), stable)

meta_new = dict(meta)
meta_new['fault_type_ids'] = {"0": "load_fault", "1": "gen_trip",
                               "2": "transmission_line_fault", "3": "transformer_fault"}
meta_new['note'] = ("Split from a merged line_fault category into transmission_line_fault "
                     "(id=2) and transformer_fault (id=3) -- see split_line_and_transformer.py "
                     "for why (5-10x severity gap between the two, corrupting shared training).")
with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
    json.dump(meta_new, f, indent=2)

print(f"\nSaved split dataset to ./{args.out_dir}/")
print(f"  X_train.npy: {X_new.shape}")
