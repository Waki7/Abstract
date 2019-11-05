AGENT_REGISTRY = {}

def register_algorithm(controller):
    AGENT_REGISTRY[controller.__name__] = controller