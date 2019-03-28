import json
import models
from keras import optimizers as keras_optimizers

# Load the config
CONFIG_PATH = './configs.json'
with open(CONFIG_PATH, 'r') as config_file:
    CONFIG = json.load(config_file)


def build_optimizer(params):
    return getattr(keras_optimizers, params.pop('name'))(**params)


def load_model(name):
    params = CONFIG[name]
    loader = getattr(models, params.pop('loader'))

    # Build the optimizer
    optim_params = params['optimizer']
    params['optimizer'] = build_optimizer(optim_params)

    return loader, params
