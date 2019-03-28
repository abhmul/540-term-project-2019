import logging
import argparse
import numpy as np

from pyjet.callbacks import Plotter
from pyjet.data import NpDataset

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


import kaggleutils as utils
from data_utils import RoadData
from augmenters import Cropper
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

# Training params
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size to use for training')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='Batch size to use for testing')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed to use')


# parser.add_argument('--num_completed', type=int, default=0, help='How many completed folds')
# parser.add_argument('--reload_model', action='store_true', help='Continues training from a saved model')
# parser.add_argument('--initial_epoch', type=int, default=0, help='Continues training from specified epoch')
# parser.add_argument('--cutoff', type=float, default=0.5, help='Cutoff to use for producing submission')
# parser.add_argument('--use_iou', action='store_true', help='creates test predictions with iou checkpointed model.')
parser.add_argument('--debug', action='store_true',
                    help='runs the script in debug mode')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_model(model,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=100,
                batch_size=32,
                val_batch_size=32,
                plot=True,
                run_id='default_model_name',
                debug=False):

    # Create the generators
    logging.info(
        f'Training model for {epochs} epochs and {batch_size} batch size')
    logging.info('Flowing the train and validation sets')
    traingen = trainset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    # traingen = Cropper(crop_size=(128, 128), augment_labels=True)(traingen)
    valgen = valset.flow(batch_size=val_batch_size, shuffle=False)

    # Create the callbacks
    logging.info('Creating the callbacks')
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(run_id),
            'val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True),
        ModelCheckpoint(
            utils.get_model_path(run_id + '_dice_coef'),
            'val_dice_coef',
            verbose=1,
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
        # Plotter(
        #     'mean_iou_t',
        #     plot_during_train=plot,
        #     save_to_file=utils.get_plot_path(RUN_ID+'_mean_iou'),
        #     block_on_end=False),
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
        verbose=1,
        max_queue_size=6)

    return logs


def test_model(model, test_data: NpDataset, batch_size=32, debug=False):
    logging.info(f'Testing model with batch size of {batch_size}')
    logging.info('Flowing the test set')
    test_data.output_labels = False
    testgen = test_data.flow(batch_size=batch_size, shuffle=False)

    test_steps = 3 if debug else testgen.steps_per_epoch

    test_preds = model.predict_generator(
        testgen,
        test_steps,
        verbose=1)

    if debug:
        filler = np.zeros(
            (len(test_data) - len(test_preds), *test_preds.shape[1:]))
        test_preds = np.concatenate([test_preds, filler])

    return test_preds.squeeze(-1)


def train(data: RoadData, model_loader, model_params, cmdargs):
    train_data = data.load_train()
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
                debug=cmdargs.debug)
    # Load the model and score it
    model.load_weights(utils.get_model_path(cmdargs.run_id))
    return model


def train_kfold(data: RoadData, model_loader, model_params, cmdargs):
    full_data = data.load_train()
    model = None
    for i, (train_data, val_data) in enumerate(full_data.kfold(cmdargs.kfold)):
        logging.info(f'Training fold {i+1}/{cmdargs.kfold}')

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
                    debug=cmdargs.debug)
    return model


def test(data: RoadData, model_loader, model_params, cmdargs, model=None):
    test_data = data.load_test()
    if model is None:
        logging.info('No model provided, constructing one.')
        model = model_loader(**model_params)
    # Load the model and score it
    model.load_weights(utils.get_model_path(cmdargs.run_id))
    test_preds = test_model(
        model,
        test_data,
        batch_size=cmdargs.test_batch_size,
        debug=cmdargs.debug
    )
    # Save the submission
    data.save_submission(
        utils.get_submission_path(cmdargs.run_id),
        test_preds,
        test_data.ids)


def test_kfold(data: RoadData, model_loader, model_params, cmdargs, model=None):
    test_data = data.load_test()
    if model is None:
        logging.info('No model provided, constructing one.')
        model = model_loader(**model_params)
    # Run all the kfold predictions
    test_preds = 0.
    for i in range(cmdargs.kfold):
        logging.info(f'Testing fold {i+1}/{cmdargs.kfold}')
        run_id = cmdargs.run_id + f'_{i}'
        model.load_weights(utils.get_model_path(run_id))
        test_preds = test_preds + \
            test_model(
                model,
                test_data,
                batch_size=cmdargs.test_batch_size,
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
    loader, params = load_model(args.run_id)

    data = RoadData()
    model = None

    # check for kfold
    if args.kfold:
        if args.train:
            model = train_kfold(data, loader, params, args)
        if args.test:
            test_kfold(data, loader, params, args, model=model)

    if args.train:
        model = train(data, loader, params, args)
    if args.test:
        test(data, loader, params, args, model=model)
