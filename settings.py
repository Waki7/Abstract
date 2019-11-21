import torch

configs = {


}

LOG_DIR = 'logs'

grads = {}
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
DTYPE = torch.float
ARGS = {'device': DEVICE, 'dtype': DTYPE}


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
