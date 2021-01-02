import logging
import os
import typing as typ

import torch

import networks.network_interface as base

NETWORK_REGISTERY: typ.Dict[str, base.NetworkInterface] = {}
IMAGE_ENCODER = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network


def try_load_weights(model: base.NetworkInterface, cfg):
    if cfg.get('pretrained', False):
        if 'load_folder' in cfg:
            model_folder = cfg['load_folder']
            if os.path.exists(model_folder):
                if hasattr(model, 'load'):  # override if model has some special loading functionality
                    model.load(model_folder)
                else:
                    logging.warning('using torch load instead of classes implementation')
                    weights_path = model.get_weights_filepath(model_folder)
                    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict)
                return True
            else:
                logging.error('path {} could not be found, cannot load pretrained blueprint_weights'.format(model_folder))
        else:
            logging.warning('path was not specified by config, cannot load pretrained blueprint_weights')
    return False


def get_network(cfg, in_shapes, out_shapes=None) -> base.NetworkInterface:
    model = NETWORK_REGISTERY[cfg['class_name']](
        cfg=cfg,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
    )
    try_load_weights(model=model, cfg=cfg)
    return model
