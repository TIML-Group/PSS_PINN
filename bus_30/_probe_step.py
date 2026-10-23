import sys; sys.argv=['t']
import os, torch, numpy as np, copy
import importlib.util
# Import refine_dp30 module pieces without running main: exec the file up to the loop is hard;
# instead inline-replicate the 30-bus step on batch0 from the real pipeline.
from case30_common import Setup, actionable, NG
su = Setup('case30_ctrl/best.pt', 'data30_mm', device='cuda')
model, A, ck = su.model, su.A, su.ck
for q in model.parameters(): q.requires_grad_(False)
tm = [model.dp_cell, model.dp_fc] if A['dp_head']=='cascade' else [model.enc.head_dP]
for m_ in tm:
    for q in m_.parameters(): q.requires_grad_(True)
trainable=[q for q in model.parameters() if q.requires_grad]
rng = np.random.default_rng(8)
ks = rng.choice(su.idx_tr, 512, replace=False)
b = su.to_dev(su.make_batch(ks, np.random.default_rng(123)))
lam = 10.0
def step(capturable, cg):
    m2 = copy.deepcopy(model) if not cg else model
    if not cg:
        for q in model.parameters(): q.requires_grad_(False)
        for m_ in tm:
            for q in m_.parameters(): q.requires_grad_(True)
    tr=[q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(tr, lr=3e-4, capturable=capturable)
    o = model(b['feat'], b['x_meas'], ctrl=True, vmask=b['vmask'])
    pk_c, has_c = model.judged(o['w_ctrl'], b['vmask'])
    viol = model.violation(o['w_ctrl'], b['vmask'])
    rm = actionable(has_c, b['x_meas'], b['active'], model.limit).float()
    p_fail = torch.sigmoid((pk_c - (model.limit-model.margin)) / A['dp_tau'])
    env = (((p_fail + A['w_viol']*viol/model.limit)*rm).sum()/rm.sum().clamp(min=1))
    nrm = o['dP'].pow(2).sum(1).mean()
    (nrm + lam*env).backward()
    torch.nn.utils.clip_grad_norm_(tr, 1.0)
    if cg:
        # replicate capture: snapshot->warmup->capture on side thread skipped; approximate:
        # capture path == same ops; just do the step (numerics = capturable adam)
        opt.step()
        return nrm.detach(), env.detach()
    opt.step()
    return nrm.detach(), env.detach()
# arm A: eager-capturable-FALSE
wa = copy.deepcopy(model.state_dict()); nA,eA = step(False, False)
sa_A = copy.deepcopy({k: v.clone() for k,v in model.state_dict().items()})
# arm B capturable TRUE (no capture)
model.load_state_dict(wa)
nB,eB = step(True, False)
sb = copy.deepcopy({k: v.clone() for k,v in model.state_dict().items()})
d = max(float((sb[k].float()-sa_A[k].float()).abs().max()) for k in sa_A)
print('grad+adam capturable-vs-not param diff after 1 step:', d)
print('nrm', float(nA), float(nB), 'env', float(eA), float(eB))
# arm C: loss values eager vs graph-side: solver safe on/off fwd diff on THIS batch:
su.S.solver_capture_safe = False
with torch.no_grad():
    o0 = model(b['feat'], b['x_meas'], ctrl=True, vmask=b['vmask'])
su.S.solver_capture_safe = True
with torch.no_grad():
    o1 = model(b['feat'], b['x_meas'], ctrl=True, vmask=b['vmask'])
print('fwd w_ctrl diff capsafe:', float((o1['w_ctrl']-o0['w_ctrl']).abs().max()))
print('fwd dP diff capsafe:', float((o1['dP']-o0['dP']).abs().max()))
