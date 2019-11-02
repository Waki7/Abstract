NETWORK_REGISTERY = {}

def register_network(controller):
    NETWORK_REGISTERY[controller.__name__] = controller