import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pyjet.callbacks import Plotter
from pyjet.data import NpDataset

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


import kaggleutils as utils
from data_utils import RoadData
from plot_utils import plot_img_grid
from augmenters import Cropper, FlipAugmenter
from config_loader import load_model

parser = argparse.ArgumentParser(description='Run the models.')
parser.add_argument('run_id', help='ID of the model configuration')
parser.add_argument('-k', '--kfold', type=int, default=0,
                    help='Run this script to train a model with kfold validation')
parser.add_argument('--train', action='store_true',
                    help='Whether to run this script to train a model')
parser.add_argument('--test', action='store_true',
                    help='Whether to run this script to generate submissions')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the training loss')
parser.add_argument('--multi_gpu', action='store_true',
                    help='Whether to use parallel gpu kfolding')

# Training params
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size to use for training')
parser.add_argument('--test_batch_size', type=int, default=4,
                    help='Batch size to use for testing')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed to use')
parser.add_argument('-ta', '--test_augment', type=int, default=0,
                    help='Number of test augmentations to use. By default does no test augmentation')

# parser.add_argument('--num_completed', type=int, default=0, help='How many completed folds')
# parser.add_argument('--reload_model', action='store_true', help='Continues training from a saved model')
# parser.add_argument('--initial_epoch', type=int, default=0, help='Continues training from specified epoch')
# parser.add_argument('--cutoff', type=float, default=0.5, help='Cutoff to use for producing submission')
# parser.add_argument('--use_iou', action='store_true', help='creates test predictions with iou checkpointed model.')
parser.add_argument('--use_np', action='store_true',
                    help='Loads the images into memory with uint8s')
parser.add_argument('--custom_weights', default='',
                    help='Custom weights to load. Only works for non-kfold')
parser.add_argument('--view_preds', action='store_true',
                    help='Displays the predictions with the input images for the test set')
parser.add_argument('--debug', action='store_true',
                    help='runs the script in debug mode')

logger = logging.getLogger()
# logger.setLevel(level='WARNING')
# logging.basicConfig(
#     format='%(asctime)s : %(levelname)s : %(message)s', level=logger.info)


def train_model(model,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=50,
                batch_size=32,
                val_batch_size=32,
                plot=True,
                run_id='default_model_name',
                augmenter=None,
                verbose=1,
                debug=False):

    # Create the generators
    logger.info(
        f'Training model for {epochs} epochs and {batch_size} batch size')
    logger.info('Flowing the train and validation sets')
    traingen = trainset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    valgen = valset.flow(batch_size=val_batch_size, shuffle=False)

    if augmenter is not None:
        logger.info(f'Training with augmenter {augmenter.image_augmenter}')
        augmenter.labels = True
        traingen = augmenter(traingen)

    # Create the callbacks
    logger.info('Creating the callbacks')
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(run_id),
            'val_loss',
            verbose=verbose,
            save_best_only=True,
            save_weights_only=True),
        ModelCheckpoint(
            utils.get_model_path(run_id + '_dice_coef'),
            'val_dice_coef',
            verbose=verbose,
            save_best_only=True,
            save_weights_only=True,
            mode='max'),
        Plotter(
            'loss',
            scale='log',
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(run_id + '_loss'),
            block_on_end=False),
        Plotter(
            'dice_coef',
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(run_id + '_dice_coef'),
            block_on_end=False),
    ]

    train_steps = 3 if debug else traingen.steps_per_epoch
    val_steps = 3 if debug else valgen.steps_per_epoch
    epochs = 2 if debug else epochs

    # Train the model
    logs = model.fit_generator(
        traingen,
        train_steps,
        epochs=epochs,
        validation_data=valgen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose,
        max_queue_size=3)

    return logs


def test_model(model, test_data: NpDataset, batch_size=32, num_augmentations=0, view_preds=False, debug=False):
    logger.info(f'Testing model with batch size of {batch_size}')
    logger.info('Flowing the test set')
    test_data.output_labels = False
    testgen = test_data.flow(batch_size=batch_size, shuffle=False)
    if num_augmentations:
        print(f'Testing with a flip augmenter')
        augmenter = FlipAugmenter(flipud=True, fliplr=True)
        aug_params = [dict(flipud=True, fliplr=True), dict(flipud=True, fliplr=False), dict(
            flipud=False, fliplr=True), dict(flipud=False, fliplr=False)]
        augmenter.labels = False
        testgen = augmenter(testgen)
    else:
        num_augmentations = 1
        augmenter = None

    test_steps = 3 if debug else testgen.steps_per_epoch

    test_preds = 0.
    for i in range(num_augmentations):
        if augmenter is not None:
            print(
                f'Testing for augmentation {i+1}/{num_augmentations} with flipud={aug_params[i]["flipud"]} and fliplr={aug_params[i]["fliplr"]}')
            augmenter.flipud = aug_params[i]['flipud']
            augmenter.fliplr = aug_params[i]['fliplr']

        aug_test_preds = model.predict_generator(
            testgen,
            test_steps,
            verbose=1,
            max_queue_size=0,
            workers=0)  # Must set to workers=0 to maintain test prediction order
        # Reverse the augmentations
        # TODO: only works with flips, implement general solution for non-flips
        if augmenter is not None:
            print('Running reverse augmentation on predictions...')
            aug_test_preds = augmenter.reverse_augment(aug_test_preds)

        if view_preds:
            if augmenter:
                testgen.generator.restart()
                display_predictions(testgen.generator, aug_test_preds)
            else:
                display_predictions(testgen, aug_test_preds)

        test_preds = test_preds + aug_test_preds
    test_preds /= num_augmentations

    if debug:
        filler = np.zeros(
            (len(test_data) - len(test_preds), *test_preds.shape[1:]))
        test_preds = np.concatenate([test_preds, filler])

    if view_preds:
        display_predictions(testgen, test_preds)

    return test_preds.squeeze(-1)


def display_predictions(testgen, predictions):
    i = 0
    for batch_x in testgen:
        for xi in batch_x:
            plot_img_grid([xi, predictions[i, ..., 0],
                           (predictions[i, ..., 0] > 0.5).astype(float)])
            i += 1


def train(data: RoadData, model_dict, cmdargs):
    train_data = data.load_train()
    model_loader, model_params = model_dict['loader'], model_dict['params']

    model = model_loader(**model_params)
    train_data, val_data = train_data.validation_split(
        split=0.1, shuffle=True, seed=utils.get_random_seed(), stratified=True,
        stratify_by=data.get_stratification_categories(train_data))
    train_model(model, train_data, val_data,
                epochs=cmdargs.epochs,
                batch_size=cmdargs.batch_size,
                val_batch_size=cmdargs.test_batch_size,
                plot=cmdargs.plot,
                run_id=cmdargs.run_id,
                augmenter=model_dict['augmenter'],
                debug=cmdargs.debug)
    # Load the model and score it
    model.load_weights(utils.get_model_path(cmdargs.run_id))
    return model


def train_kfold(data: RoadData, model_dict, cmdargs):
    full_data = data.load_train()
    model_loader, model_params = model_dict['loader'], model_dict['params']

    model = None
    for i, (train_data, val_data) in enumerate(full_data.kfold(cmdargs.kfold)):
        logger.info(f'Training fold {i+1}/{cmdargs.kfold}')

        # Clean up the previous model and make a new one
        del model
        model = model_loader(**model_params)

        run_id = cmdargs.run_id + f'_{i}'
        train_model(model, train_data, val_data,
                    epochs=cmdargs.epochs,
                    batch_size=cmdargs.batch_size,
                    val_batch_size=cmdargs.test_batch_size,
                    plot=cmdargs.plot,
                    run_id=run_id,
                    augmenter=model_dict['augmenter'],
                    debug=cmdargs.debug)
    return model


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def train_gpu(gpu, model_dict, run_id, train_data, val_data, cmdargs):
    model_loader, model_params = model_dict['loader'], model_dict['params']
    # with tf.Session(graph=tf.Graph()) as sess:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # K.clear_session()
        K.set_session(sess)
        with tf.device(gpu):
            model = model_loader(**model_params)
            train_model(model, train_data, val_data,
                        epochs=cmdargs.epochs,
                        batch_size=cmdargs.batch_size,
                        val_batch_size=cmdargs.test_batch_size,
                        plot=cmdargs.plot,
                        run_id=run_id,
                        augmenter=model_dict['augmenter'],
                        debug=cmdargs.debug,
                        verbose=2)
            return True


def train_parallel_kfold(executor, data: RoadData, model_dict, cmdargs):
    full_data = data.load_train()

    gpus = get_available_gpus()
    logger.info(f'Found {len(gpus)} GPU(s).')

    fold_futures = [None for _ in range(cmdargs.kfold)]
    fold_to_gpu = {}
    fold_queue = deque()

    for i, (train_data, val_data) in enumerate(full_data.kfold(cmdargs.kfold)):
        if len(gpus) == 0:
            # We are out of gpus, so need to wait for the next one
            first_fold = fold_queue.pop()
            first_gpu = fold_to_gpu.pop(first_fold)
            assert fold_futures[first_fold] is not None
            print(
                f'Fold {i} waiting on fold {first_fold} running on device {first_gpu}...')
            fold_futures[first_fold].result()
            print(f'Fold {first_fold} completed.')
            # The training is done
            gpus.append(first_gpu)
            fold_futures[first_fold] = None

        next_gpu = gpus.pop()  # Get the next gpu
        run_id = cmdargs.run_id + f'_{i}'
        print(f'Running fold {i} on device {next_gpu}...')
        # Create the future
        fold_futures[i] = executor.submit(
            train_gpu, next_gpu, model_dict, run_id, train_data, val_data, cmdargs)
        # Map the fold to the gpu and add the fold to the queue
        fold_to_gpu[i] = next_gpu
        fold_queue.appendleft(i)

    # Wait for the remaining futures
    for future in fold_futures:
        if future is not None:
            future.result()

    return None


def test(data: RoadData, model_dict, cmdargs, model=None):
    test_data = data.load_test()
    model_loader, model_params = model_dict['loader'], model_dict['params']
    augmenter = None
    if cmdargs.test_augment:
        num_augmentations = cmdargs.test_augment
        augmenter = model_dict['augmenter']
    else:
        num_augmentations = 0
        augmenter = None
    if model is None:
        logger.info('No model provided, constructing one.')
        model = model_loader(**model_params)
    # Load the model and score it
    if cmdargs.custom_weights:
        print(f'Loading custom weights from {cmdargs.custom_weights}')
        model.load_weights(cmdargs.custom_weights)
    else:
        model.load_weights(utils.get_model_path(cmdargs.run_id))
    test_preds = test_model(
        model,
        test_data,
        batch_size=cmdargs.test_batch_size,
        # augmenter=augmenter,
        num_augmentations=num_augmentations,
        view_preds=cmdargs.view_preds,
        debug=cmdargs.debug
    )
    # Save the submission
    data.save_submission(
        utils.get_submission_path(cmdargs.run_id),
        test_preds,
        test_data.ids)


def test_kfold(data: RoadData, model_dict, cmdargs, model=None):
    test_data = data.load_test()
    model_loader, model_params = model_dict['loader'], model_dict['params']
    augmenter = None
    if cmdargs.test_augment:
        num_augmentations = cmdargs.test_augment
        augmenter = model_dict['augmenter']
    else:
        num_augmentations = 0
        augmenter = None
    if model is None:
        logger.info('No model provided, constructing one.')
        model = model_loader(**model_params)
    # Run all the kfold predictions
    test_preds = 0.
    for i in range(cmdargs.kfold):
        print(f'Testing fold {i+1}/{cmdargs.kfold}')
        run_id = cmdargs.run_id + f'_{i}'
        model.load_weights(utils.get_model_path(run_id))
        test_preds = test_preds + \
            test_model(
                model,
                test_data,
                batch_size=cmdargs.test_batch_size,
                # augmenter=augmenter,
                num_augmentations=num_augmentations,
                view_preds=cmdargs.view_preds,
                debug=cmdargs.debug
            )
    # Average the predictions
    test_preds /= cmdargs.kfold
    # Save the submission
    data.save_submission(
        utils.get_submission_path(cmdargs.run_id),
        test_preds,
        test_data.ids)


if __name__ == '__main__':
    args = parser.parse_args()

    # Set the random seed
    utils.set_random_seed(args.seed)

    # Get the model loader and params
    model_dict = load_model(args.run_id)

    data = RoadData(train_np=args.use_np, test_np=args.use_np)
    model = None

    # check for kfold
    if args.kfold:
        if args.train:
            if args.multi_gpu:
                with ThreadPoolExecutor(max_workers=args.kfold) as executor:
                    train_parallel_kfold(executor, data, model_dict, args)
        if args.test:
            test_kfold(data, model_dict, args, model=model)
    else:
        if args.train:
            model = train(data, model_dict, args)
        if args.test:
            test(data, model_dict, args, model=model)
