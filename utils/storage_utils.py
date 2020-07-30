import yaml

def save_config(cfg, path):
    with open(path, 'w') as file:
        yaml.dump(cfg, file)


def load_config(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg