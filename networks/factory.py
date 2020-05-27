NETWORK_REGISTERY = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network


def get_network(key, cfg, in_shapes, out_shapes):
    return NETWORK_REGISTERY[key](
        cfg=cfg,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
    )
