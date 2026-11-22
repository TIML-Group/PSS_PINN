"""Q7: evaluate the frozen generalist baseline (pure pole placement, seeds_v2.json
'generalist') on a representative era-1-mix sample. Writes baseline_generalist.json."""
import json, sys, numpy as np, torch
from plant_v2 import PlantV2, const_gains, LIMIT, actuator_hash
from scenarios_v2 import draw
dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
SRC = sys.argv[2] if len(sys.argv) > 2 else 'pole'      # 'pole' (seeds_v2.json) | 'ceiling' (baseline_ceiling.json)
N, SEED, ACCEPT = 1200, 7, 0.85
p = PlantV2(device=dev); g = np.array(json.load(open('seeds_v2.json'))['generalist'] if SRC == 'pole' else json.load(open('baseline_ceiling.json'))['gains'])
S = draw(p, N, SEED); ty = np.array([s['type'] for s in S])
o = p.simulate(p.batch(S), const_gains(N, p.n_lbl, g, p.device, p.dtype))
pk = o['peak'].cpu().numpy(); we = (o['w_end'] - 1).abs().amax(1).cpu().numpy()
r = dict(actuator_hash=actuator_hash(), n=N, seed=SEED, acceptance_threshold=ACCEPT,
         in_envelope=float((pk <= LIMIT).mean()),
         by_type={t: dict(n=int((ty == t).sum()), in_envelope=float((pk[ty == t] <= LIMIT).mean())) for t in ['gen', 'load', 'line']},
         settled_1e3_at_8s=float((we < 1e-3).mean()), median_peak=float(np.median(pk)))
r['accepted'] = r['in_envelope'] >= ACCEPT
r['gains_source'] = SRC
r['by_type_alpha'] = {t: {b: float((pk[(ty == t) & m] <= LIMIT).mean()) for b, m in (('a<0.4', np.array([s['alpha'] < 0.4 for s in S])), ('0.4-0.7', np.array([0.4 <= s['alpha'] < 0.7 for s in S])), ('a>=0.7', np.array([s['alpha'] >= 0.7 for s in S]))) if ((ty == t) & m).any()} for t in ['gen', 'load']}
r['gen_by_machine'] = {m: float((pk[(ty == 'gen') & (np.array([s['loc'] for s in S]) == m)] <= LIMIT).mean()) for m in (1, 2)}
r['max_dev_median_by_type'] = {t: float(np.median(pk[ty == t])) for t in ['gen', 'load', 'line']}
json.dump(r, open('baseline_generalist.json' if SRC == 'pole' else 'baseline_ceiling_eval.json', 'w'), indent=1); print(json.dumps(r, indent=1))
