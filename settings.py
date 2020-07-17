import logging
import random

import numpy as np
import torch

LOG_DIR = 'logs'
ENCODER_WEIGHTS = 'networks/weights'

grads = {}
if torch.cuda.is_available():
    logging.info('using cpu')
    DEVICE = torch.device('cuda')
else:
    DEVICE_NUM = 0
    logging.info('{} gpus available, will be using gpu {}'.format(torch.cuda.device_count(), DEVICE_NUM))
    torch.cuda.set_device(DEVICE_NUM)
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
