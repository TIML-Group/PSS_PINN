import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import numpy as np
import torch
from case30_common import Setup, NG

device = torch.device('cpu')
su = Setup('case30_ctrl/refined_full/best.pt', 'data30_mm', device=device)
S = su.S
print("x0 omega slice:", S.x0[NG:2*NG].tolist())
print("x0 full head:", S.x0[:20].tolist())
