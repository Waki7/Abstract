AGENT_REGISTRY = {}


def register_agent(algorithm):
    AGENT_REGISTRY[algorithm.__name__] = algorithm
    return algorithm
