import logging
import os
from typing import Dict

import torch

import networks.network_interface as base

NETWORK_REGISTERY: Dict[str, base.NetworkInterface] = {}
IMAGE_ENCODER = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network


def try_load_weights(model, cfg):
    if cfg.get('pretrained', False):
        if hasattr(model, 'weights_path'):
            weights_path = model.weights_path
            if hasattr(model, 'load'):  # override if model has some special loading functionality
                model.load(model.weights_path)
                return True
            if os.path.exists(model.weights_path):
                state_dict = torch.load(weights_path)
                model.load_state_dict(state_dict)
                return True
            else:
                logging.error('path {} could not be found, cannot load pretrained weights'.format(weights_path))
        else:
            logging.warning('path was not specified by model, cannot load pretrained weights')
    return False


def get_network(key, cfg, in_shapes, out_shapes) -> base.NetworkInterface:
    model = NETWORK_REGISTERY[key](
        cfg=cfg,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
    )
    try_load_weights(model=model, cfg=cfg)
    return model
