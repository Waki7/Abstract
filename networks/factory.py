NETWORK_REGISTERY = {}


def register_network(network):
    NETWORK_REGISTERY[network.__name__] = network
    return network
