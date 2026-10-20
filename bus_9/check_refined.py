"""
Did refinement touch anything other than the dP head?

Refinement freezes every parameter except dp_cell/dp_fc, so the encoder tensors
in refined/best.pt must be bit-identical to the ones in best.pt. If they are
not, the refined checkpoint did not come from the patched refine_dp.py, or
something else is leaking.

    python check_refined.py receding_cascade
"""
import sys, os, torch

run = sys.argv[1] if len(sys.argv) > 1 else 'receding_cascade'
orig_p = os.path.join(run, 'best.pt')
ref_p = os.path.join(run, 'refined', 'best.pt')

a = torch.load(orig_p, map_location='cpu', weights_only=False)
b = torch.load(ref_p, map_location='cpu', weights_only=False)

print(f"original : {orig_p}   epoch {a.get('epoch')}")
print(f"refined  : {ref_p}    refined_from {b.get('refined_from')}")
print()

# ── 1. which script produced the refined checkpoint? ────────────────────────
ra = b.get('refine_args') or {}
has_warmup = 'warmup' in ra
print("PROVENANCE")
print(f"  refine_args has 'warmup' : {has_warmup}"
      f"   -> {'PATCHED script' if has_warmup else 'OLD script (pre-fix)'}")
if ra:
    for k in ('warmup', 'floor_tol', 'rescue_floor', 'patience', 'lam0'):
        if k in ra:
            print(f"    {k:14s} {ra[k]}")
print()

# ── 2. beta ─────────────────────────────────────────────────────────────────
sa, sb = a['model'], b['model']
print("BETA (mask sharpness)")
if 'enc.beta' in sa:
    ba, bb = float(sa['enc.beta']), float(sb['enc.beta'])
    bmin = a.get('args', {}).get('beta_min')
    print(f"  original {ba:.6f}   refined {bb:.6f}   beta_min {bmin}")
    if abs(ba - bb) > 0:
        print(f"  ** BETA CHANGED by {abs(ba-bb):.6g} -- this is the bug. **")
    else:
        print("  unchanged -- correct")
print()

# ── 3. exactly which tensors differ ─────────────────────────────────────────
print("TENSORS THAT CHANGED")
changed, same = [], 0
for k in sa:
    if k not in sb:
        print(f"  MISSING in refined: {k}")
        continue
    ta, tb = sa[k], sb[k]
    if ta.shape != tb.shape:
        changed.append((k, float('nan')))
        continue
    if not ta.dtype.is_floating_point:
        continue
    d = (ta.float() - tb.float()).abs().max().item()
    if d > 0:
        changed.append((k, d))
    else:
        same += 1

expected = ('dp_cell', 'dp_fc')
leaks = [(k, d) for k, d in changed if not any(e in k for e in expected)]

for k, d in changed:
    tag = '  (expected)' if any(e in k for e in expected) else '  <-- LEAK'
    print(f"  {k:42s} max|diff| {d:.3e}{tag}")
print(f"\n  {same} tensors identical, {len(changed)} changed, {len(leaks)} unexpected")
print()

print("VERDICT")
if leaks:
    print("  FAIL: tensors outside the dP head changed. The encoder is not the")
    print("        one that was trained, so identification metrics will differ.")
    if not has_warmup:
        print("        Cause is almost certainly the OLD refine script -- re-run")
        print("        refinement with the patched refine_dp.py.")
elif not changed:
    print("  The two checkpoints are IDENTICAL. Refinement produced no change at")
    print("  all -- check that the refine run actually wrote to this path.")
else:
    print("  PASS: only dp_cell/dp_fc changed. Identification is untouched, so")
    print("  any difference in accuracy/alpha/TPR between before and after is")
    print("  coming from the evaluation, not the checkpoint.")
