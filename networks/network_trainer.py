import os
import typing as typ

import torch
import torch.nn as nn

import settings
import utils.model_utils as model_utils
from utils.paths import Networks


class NetworkTrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.updates_locked = False
        self.gradient_clip = cfg.get('gradient_clip', settings.defaults.GRADIENT_CLIP)
        self.optimizer = None
        self.params = None

    def init_optimizer(self, parameters: typ.Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        # floating point precision, so need to set epislon
        return getattr(torch.optim, optimizer)(parameters, lr=lr, eps=1.e-4)

    def add_network(self, network: nn.Module):
        self.params = list(network.parameters())
        self.optimizer: torch.optim.Optimizer = self.init_optimizer(self.params)
        model_utils.module_dtype_init(network)

    def add_layer_to_optimizer(self, layer: nn.Module):
        self.params = list(layer.parameters()) + list(self.params)
        self.optimizer = self.init_optimizer(self.params)
        model_utils.module_dtype_init(layer)

    def lock_updates(self):
        self.updates_locked = True

    def unlock_updates(self):
        self.updates_locked = False

    def update_parameters(self, override_lock=False):
        if (not self.updates_locked) or override_lock:
            torch.nn.utils.clip_grad_value_(self.params, self.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def store_optimizer(self, model_folder):
        # todo, maybe load optimizer?
        optimizer_filename = os.path.join(model_folder, Networks.OPTIMIZER_FILENAME)
        torch.save(self.optimizer.state_dict(), optimizer_filename)

    def save(self, model_folder):
        self.store_optimizer(model_folder)
