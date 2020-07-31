import os

import torch
import torch.nn as nn

import settings


class NetworkTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.updates_locked = False
        self.gradient_clip = cfg.get('gradient_clip', settings.defaults.GRADIENT_CLIP)
        self.OPTIMIZER_FILENAME = 'optimizer.pth'
        self.optimizer = None

    def create_optimizer(self, network: nn.Module):
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        # floating point precision, so need to set epislon
        self.optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer)(network.parameters(), lr=lr, eps=1.e-4)
        settings.device_init(network)

    def add_layer(self, layer: nn.Module):
        settings.device_init(layer)
        self.optimizer.param_groups.append({'params': layer.parameters()})

    def lock_updates(self):
        self.updates_locked = True

    def unlock_updates(self):
        self.updates_locked = False

    def update_parameters(self, override_lock=False):
        if (not self.updates_locked) or override_lock:
            torch.nn.utils.clip_grad_value_(self.optimizer.param_groups, self.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_optimizer_filename(self, model_folder):
        return os.path.join(model_folder, self.OPTIMIZER_FILENAME)

    def load(self, model_folder):
        pass
        # model.load_state_dict(torch.load(PATH, map_location=device))

    def store_optimizer(self, model_folder):
        torch.save(self.optimizer.state_dict(), self.get_optimizer_filename(model_folder))

    def save(self, model_folder):
        self.store_optimizer(model_folder)
