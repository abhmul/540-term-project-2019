import json
import models
from keras import optimizers as keras_optimizers
from imgaug import augmenters as iaa
from augmenters import ImageAugmenter

# Load the config
CONFIG_PATH = './configs.json'
with open(CONFIG_PATH, 'r') as config_file:
    CONFIG = json.load(config_file)
TRAIN_CONFIG_PATH = './train_configs.json'
with open(TRAIN_CONFIG_PATH, 'r') as config_file:
    TRAIN_CONFIG = json.load(config_file)


def build_augmenter(params):
    augmenters = []
    for p in params:
        augmenters.append(getattr(iaa, p.pop('name'))(**p))
    image_augmenter = iaa.Sequential(augmenters, random_order=True)
    return ImageAugmenter(image_augmenter, labels=True, augment_labels=True)


def load_model(name):
    params = CONFIG[name]
    return_dict = {}
    return_dict['loader'] = getattr(models, params.pop('loader'))

    # Build the optimizer
    optim_params = params['optimizer']
    params['optimizer'] = optim_params

    # Build the augmenter if there is one
    if 'augmenter' in params:
        augmenter = build_augmenter(params.pop('augmenter'))
        return_dict['augmenter'] = augmenter

    return_dict['params'] = params

    return return_dict


def load_train(name, cmdargs):
    params = TRAIN_CONFIG[name]
    for k, v in params.items():
        assert hasattr(cmdargs, k), f'Found invalid cmdline argument "{k}"'
        setattr(cmdargs, k, v)
    return cmdargs
