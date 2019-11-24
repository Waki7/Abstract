NETWORK_REGISTERY = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network


def get_network(key, cfg, out_shape, out_shape2 = None, n_features=0, in_image_shape=None):
    return NETWORK_REGISTERY[key](
        cfg=cfg,
        out_shape=out_shape,
        out_shape2=out_shape2,
        n_features=n_features,
        in_image_shape = in_image_shape
    )
