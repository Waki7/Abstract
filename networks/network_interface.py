import logging
import os

import torch
import torch.nn as nn

import settings
import utils.model_utils as model_utils
import utils.storage_utils as storage_utils
from networks.network_trainer import NetworkTrainer


class NetworkInterface(nn.Module):
    def __init__(self, in_shapes, out_shapes, cfg={}):
        super(NetworkInterface, self).__init__()
        if isinstance(in_shapes, tuple):
            in_shapes = [in_shapes, ]
            logging.warning('please pass in your shapes as a list of tuples as if the network input is multi modal')
        if isinstance(out_shapes, tuple):
            out_shapes = [out_shapes, ]
            logging.warning('please pass in your shapes as a list of tuples as if the netowrk output is multi modal')
        self.cfg = {} if not hasattr(self, 'cfg') else self.cfg
        self.extra_parameters = nn.ParameterList()
        self.in_shapes = in_shapes
        self.out_shapes = out_shapes
        self.in_features = model_utils.sum_multi_modal_shapes(in_shapes)
        self.out_features = model_utils.sum_multi_modal_shapes(out_shapes)
        self.pretrained = cfg.get('pretrained')

        self.CONFIG_FILENAME = 'config.yaml'
        self.WEIGHTS_FILENAME = 'model.pth'

        self.trainer = NetworkTrainer(self.cfg)
        self.temp_classifier = nn.Identity()

    def create_optimizer(self):
        self.trainer.add_network(self)
        return self.trainer

    def forward(self, *input):
        raise NotImplementedError

    def pretrain(self, layer):
        self.temp_classifier = layer
        self.trainer.add_layer_to_optimizer(layer)

    def get_in_shapes(self):
        return self.in_shapes

    def get_out_shapes(self):
        return self.out_shapes

    def get_out_features(self):
        return self.out_features

    def get_in_features(self):
        return self.in_features

    def get_config_filename(self, model_folder):
        return os.path.join(model_folder, self.CONFIG_FILENAME)

    def store_config(self, model_folder):
        storage_utils.save_config(self.cfg, self.get_config_filename(model_folder))

    def load_config(self, model_folder):
        return storage_utils.load_config(self.get_config_filename(model_folder))

    def get_weights_filename(self, model_folder):
        return os.path.join(model_folder, self.WEIGHTS_FILENAME)

    def store_weights(self, model_folder):
        torch.save(self.state_dict(), self.get_weights_filename(model_folder))

    def load(self, model_folder):
        self.cfg = self.load_config(model_folder)
        self.load_state_dict(torch.load(self.get_weights_filename(model_folder), map_location=settings.DEVICE))

    def save(self, model_folder):
        self.temp_classifier = nn.Identity()
        self.store_optimizer(model_folder)
        self.store_weights(model_folder)
        self.store_config(model_folder)

