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