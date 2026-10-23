import sys; sys.argv=['t']
import torch, numpy as np
from case30_physics import CASE30, NG, NB
S = CASE30('case30_base.mat', 'case30_simulink_data.mat', device='cpu', dtype=torch.float32)
dY, is_load, is_gen, gen_idx = S.dY_patterns()
torch.manual_seed(3)
B = 256
alpha0 = 0.1 + 0.9*torch.rand(B)
def f(flag):
    S.solver_capture_safe = flag
    a = alpha0.clone().requires_grad_(True)
    Y = S.Y_pre.unsqueeze(0) + a.view(-1,1,1).to(S.cdtype)*dY[1].unsqueeze(0)
    Yr, Vr = S.kron_reduce(Y)
    loss = Yr.abs().pow(2).sum()
    loss.backward()
    return Yr.detach(), Vr.detach(), loss.detach(), a.grad.clone()
Y0,V0,l0,g0 = f(False); Y1,V1,l1,g1 = f(True)
print('f32 Yred  agreement:', float((Y1-Y0).abs().max()), '| rel', float((Y1-Y0).abs().max()/Y0.abs().max()))
print('f32 Vrec  agreement:', float((V1-V0).abs().max()))
print('f32 loss  agreement:', float((l1-l0).abs()))
print('f32 grad  agreement:', float((g1-g0).abs().max()), '| rel', float((g1-g0).abs().max()/g0.abs().max()))
