CONTROLLER_REGISTERY = {}

def register_controller(controller):
    CONTROLLER_REGISTERY[controller.__name__] = controller