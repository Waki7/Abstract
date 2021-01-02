import logging
import os
from typing import *

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

        self.trainer: Optional[NetworkTrainer] = None
        self.temp_predictor = nn.Identity()

    def create_optimizer(self) -> NetworkTrainer:
        self.trainer = NetworkTrainer(self.cfg)
        self.trainer.add_network(self)
        return self.trainer

    def forward(self, *input):
        raise NotImplementedError

    def add_temp_predictor(self, predictor):
        self.temp_predictor = predictor
        self.trainer.add_layer_to_optimizer(predictor)

    def get_in_shapes(self):
        return self.in_shapes

    def get_out_shapes(self):
        return self.out_shapes

    def get_out_features(self):
        return self.out_features

    def get_in_features(self):
        return self.in_features

    # loading and saving

    def get_config_filename(self, model_folder):
        return os.path.join(model_folder, self.CONFIG_FILENAME)

    def store_config(self, model_dir_path):
        storage_utils.save_config(self.cfg, self.get_config_filename(model_dir_path))

    def load_config(self, model_dir_path):
        return storage_utils.load_config(self.get_config_filename(model_dir_path))

    def get_weights_filepath(self, model_dir_path):
        return os.path.join(model_dir_path, self.WEIGHTS_FILENAME)

    def store_weights(self, model_folder):
        torch.save(self.state_dict(), self.get_weights_filepath(model_folder))

    def load(self, load_folder):
        self.cfg = self.load_config(load_folder)
        self.load_state_dict(torch.load(self.get_weights_filepath(load_folder), map_location=settings.DEVICE))

    def save(self, save_folder):
        self.temp_predictor = nn.Identity()
        self.trainer.store_optimizer(save_folder)
        self.store_weights(save_folder)
        self.store_config(save_folder)
