import logging
import os

import torch

NETWORK_REGISTERY = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network


def get_network(key, cfg, in_shapes, out_shapes):
    model = NETWORK_REGISTERY[key](
        cfg=cfg,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
    )
    if cfg.get('from_pretrain', False):
        if hasattr(model, 'weights_path'):
            weights_path = model.weights_path
            if os.path.exists(model.weights_path):
                model.load_state_dict(torch.load(weights_path))
            else:
                logging.error('path {} could not be found, cannot load pretrained weights'.format(weights_path))
        else:
            logging.warning('path was not specified by model, cannot load pretrained weights')
    return model
