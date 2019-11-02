ALGORITHM_REGISTRY = {}

def register_algorithm(controller):
    ALGORITHM_REGISTRY[controller.__name__] = controller