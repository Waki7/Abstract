import random

import numpy as np
import torch

configs = {

}

LOG_DIR = 'logs'

grads = {}
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

DTYPE_LONG = torch.long
DTYPE_X = torch.half  # torch.float torch.half
ARGS = {'device': DEVICE, 'dtype': DTYPE_X}

SEED = 23
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class defaults:
    MODEL_SIZE = 32
    TD_STEP = 10
    LR = .001
    OPTIMIZER = 'Adam'
    DISCOUNT_FACTOR = .99
    ENTROPY_COEF = .05
    GRADIENT_CLIP = 2.0
