import torch
import torch.nn as nn

import settings
import utils.model_utils as model_utils


class NetworkInterface(nn.Module):
    def __init__(self, in_shapes, out_shapes, cfg={}):
        super(NetworkInterface, self).__init__()
        self.cfg = {}
        self.extra_parameters = nn.ParameterList()
        self.in_shapes = in_shapes
        self.in_features = model_utils.sum_multi_modal_shapes(in_shapes)
        self.out_features = model_utils.sum_multi_modal_shapes(out_shapes)
        self.weights_path = cfg.get('weights_path')
        self.pretrained = cfg.get('pretrained')
        self.extra_parameters = nn.ParameterList()
        self.updates_locked = False

        self.optimizer = None  # call create_optimizer at end of your implementation's init

    def create_optimizer(self):
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        # floating point precision, so need to set epislon
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr, eps=1.e-4)
        self.to(settings.DEVICE)
        self.half()  # convert to half precision
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    def add_parameters(self, parameters):
        self.extra_parameters.extend(parameters)
        self.create_optimizer()  # recreate optimizer due to neew parameters

    def update_parameters(self, override_lock=False):
        if (not self.updates_locked) or override_lock:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def lock_updates(self):
        self.updates_locked = True

    def unlock_updates(self):
        self.updates_locked = False
