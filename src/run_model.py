import logging
import argparse
import numpy as np

from pyjet.callbacks import Plotter
from pyjet.data import NpDataset

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


import kaggleutils as utils
from data_utils import RoadData
import models
from augmenters import Cropper

parser = argparse.ArgumentParser(description='Run the models.')
# parser.add_argument('train_id', help='ID of the train configuration')
parser.add_argument('-k', '--kfold', type=int, default=0,
                    help='Run this script to train a model with kfold validation')
parser.add_argument('--train', action='store_true',
                    help='Whether to run this script to train a model')
parser.add_argument('--test', action='store_true',
                    help='Whether to run this script to generate submissions')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the training loss')
# parser.add_argument('--num_completed', type=int, default=0, help='How many completed folds')
# parser.add_argument('--reload_model', action='store_true', help='Continues training from a saved model')
# parser.add_argument('--initial_epoch', type=int, default=0, help='Continues training from specified epoch')
# parser.add_argument('--cutoff', type=float, default=0.5, help='Cutoff to use for producing submission')
# parser.add_argument('--use_iou', action='store_true', help='creates test predictions with iou checkpointed model.')
# parser.add_argument('--test_debug', action='store_true', help='debugs the test output.')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MODEL = models.get_dilated_unet
MODEL_PARAMS = dict(optimizer=Adam(lr=0.01), filters=16)
RUN_ID = 'unet-dilated-2'
SEED = 42
BATCH_SIZE = 64
TEST_BATCH_SIZE = 4
EPOCHS = 100
utils.set_random_seed(SEED)
SPLIT_SEED = utils.get_random_seed()


def train_model(model,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                val_batch_size=TEST_BATCH_SIZE,
                plot=True,
                run_id=RUN_ID):

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

    # Train the model
    logs = model.fit_generator(
        traingen,
        traingen.steps_per_epoch,
        epochs=epochs,
        validation_data=valgen,
        validation_steps=valgen.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
        max_queue_size=6)

    return logs


def test_model(model, test_data: NpDataset, batch_size=TEST_BATCH_SIZE):
    logging.info(f'Testing model with batch size of {batch_size}')
    logging.info('Flowing the test set')
    test_data.output_labels = False
    testgen = test_data.flow(batch_size=batch_size, shuffle=False)
    test_preds = model.predict_generator(
        testgen, testgen.steps_per_epoch, verbose=1)
    return test_preds.squeeze(-1)


def train(data: RoadData, plot=True):
    train_data = data.load_train()
    model = MODEL(**MODEL_PARAMS)
    train_data, val_data = train_data.validation_split(
        split=0.1, shuffle=True, seed=SPLIT_SEED, stratified=True,
        stratify_by=data.get_stratification_categories(train_data))
    train_model(model, train_data, val_data, plot=plot)
    # Load the model and score it
    model.load_weights(utils.get_model_path(RUN_ID))
    return model


def train_kfold(data: RoadData, k, plot=True, main_run_id=RUN_ID):
    full_data = data.load_train()
    model = MODEL(**MODEL_PARAMS)
    for i, (train_data, val_data) in enumerate(full_data.kfold(k)):
        run_id = main_run_id + f'_{i}'
        train_model(model, train_data, val_data, plot=plot, run_id=run_id)
    return model


def test(data: RoadData, model=None, run_id=RUN_ID):
    test_data = data.load_test()
    if model is None:
        logging.info('No model provided, constructing one.')
        model = MODEL(**MODEL_PARAMS)
    # Load the model and score it
    model.load_weights(utils.get_model_path(run_id))
    test_preds = test_model(model, test_data)
    # Save the submission
    data.save_submission(
        utils.get_submission_path(run_id),
        test_preds,
        test_data.ids)


def test_kfold(data: RoadData, k, model=None, main_run_id=RUN_ID):
    test_data = data.load_test()
    if model is None:
        logging.info('No model provided, constructing one.')
        model = MODEL(**MODEL_PARAMS)
    # Run all the kfold predictions
    test_preds = 0.
    for i in range(k):
        logging.info(f'Testing fold {i+1}/{k}')
        run_id = main_run_id + f'_{i}'
        model.load_weights(utils.get_model_path(run_id))
        test_preds = test_preds + test_model(model, test_data)
    # Average the predictions
    test_preds /= k
    # Save the submission
    data.save_submission(
        utils.get_submission_path(main_run_id),
        test_preds,
        test_data.ids)


if __name__ == '__main__':
    args = parser.parse_args()
    data = RoadData()
    model = None

    # check for kfold
    if args.k:
        if args.train:
            model = train_kfold(data, args.k, plot=args.plot)
        if args.test:
            test_kfold(data, args.k, model=model)

    if args.train:
        model = train(data, plot=args.plot)
    if args.test:
        test(data, model=model)
