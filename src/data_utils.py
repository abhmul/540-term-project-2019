import logging
import os
from glob import glob
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize

from pyjet.data import ImageMaskDataset, ImageDataset

from kaggleutils import dump_args
from plot_utils import plot_img_grid

# All images should be 512 x 512
ORIG_IMG_SIZE = (512, 512)
IMG_SIZE = (128, 128)
EMPTY_THRESHOLD = 5
MASK_SUFFIX = '_msk'
IMG_SUFFIX = '_sat'


class ImageNPDataset(ImageDataset):

    def __init__(self, x, y=None, ids=None, img_size=None, mode="rgb"):
        super(ImageNPDataset, self).__init__(x, y=y, ids=ids)
        self.img_size = img_size
        assert mode in ImageDataset.MODE2FUNC, "Invalid mode %s" % mode
        self.mode = mode
        logging.info(
            "Creating ImageDataset(img_size={img_size}, mode={mode}".format(
                img_size=self.img_size, mode=self.mode))
        logging.info('Loading the images to uint8')
        self.x = self.load_all_imgs()
        print(self.x.shape)

    def load_all_imgs(self):
        x = self.load_img_batch(
            self.x, img_size=self.img_size, mode=self.mode, to_float=False)[0]
        return x

    def create_batch(self, batch_indicies):
        batch_x = self.x[batch_indicies].astype(np.float32)
        if self.output_labels:
            return batch_x, self.y[batch_indicies]
        return batch_x


class ImageRLEDataset(ImageDataset):

    @staticmethod
    def load_rle_batch(rles, img_size=None):
        masks = []
        for rle in rles:
            mask = rle_decoding(rle, img_size=ORIG_IMG_SIZE)
            if img_size is not None and img_size != ORIG_IMG_SIZE:
                mask = resize(
                    mask, img_size, mode='constant',
                    preserve_range=True).astype(np.float32)
            masks.append(mask)

        # If variable image size, stack the images as numpy arrays otherwise
        # create one large numpy array
        if img_size is None:
            masks = np.array(masks, dtype='O')
        else:
            masks = np.stack(masks)

        return masks

    def create_batch(self, batch_indicies):
        filenames = self.x[batch_indicies]
        x = self.load_img_batch(
            filenames, img_size=self.img_size, mode=self.mode)[0]

        if self.output_labels:
            return x, self.load_rle_batch(self.y[batch_indicies], img_size=self.img_size)
        return x


class ImageNPRLEDataset(ImageNPDataset, ImageRLEDataset):

    def create_batch(self, batch_indicies):
        batch_x = self.x[batch_indicies].astype(np.float32)
        if self.output_labels:
            return batch_x, self.load_rle_batch(self.y[batch_indicies], img_size=self.img_size)
        return batch_x


class RoadData(object):
    @dump_args
    def __init__(self, path_to_data='../input', mode='rgb',
                 img_size=ORIG_IMG_SIZE, train_np=False, test_np=False):
        self.path_to_data = path_to_data
        self.mode = mode
        self.img_size = img_size
        self.train_np = train_np
        self.test_np = test_np

        self.path_to_train_images = os.path.join(path_to_data, 'train')
        self.path_to_test_images = os.path.join(path_to_data, 'val')
        self.path_to_train_rle = os.path.join(path_to_data, 'train.csv')

        self.glob_train_images = os.path.join(
            self.path_to_train_images, f'*{IMG_SUFFIX}.jpg')
        self.glob_train_masks = os.path.join(
            self.path_to_train_images, f'*{MASK_SUFFIX}.png')
        self.glob_test_images = os.path.join(
            self.path_to_test_images, f'*{IMG_SUFFIX}.jpg')

    @staticmethod
    def get_img_id(img_path):
        img_basename = os.path.basename(img_path)
        img_id = os.path.splitext(img_basename)[0][:-len(IMG_SUFFIX)]
        return img_id

    def load_train(self):
        logging.info(f'Loading train images from {self.path_to_train_images}')
        img_paths = sorted(glob(self.glob_train_images),
                           key=lambda p: int(self.get_img_id(p)))
        # Read the rle csv
        train_rle = pd.read_csv(self.path_to_train_rle, index_col=0)

        # Initialize data containers
        ids = []
        x = []
        y = []
        for i, img_path in enumerate(tqdm(img_paths)):
            # Get the image id
            img_id = self.get_img_id(img_path)
            # Get the mask rle
            y.append(ast.literal_eval(
                train_rle.loc[int(img_id)].EncodedPixels))

            # Add the id
            ids.append(img_id)
            x.append(img_path)

        # Turn into numpy arrays
        x = np.array(x)
        y = np.array(y)
        print('X shape:', x.shape)
        print('Y Shape:', y.shape)
        if self.train_np:
            print('Using in memory image training dataset')
            return ImageNPRLEDataset(x, y=y, ids=np.array(ids))
        return ImageRLEDataset(x, y=y, ids=np.array(ids))

    def load_test(self):
        logging.info(f'Loading test images from {self.path_to_test_images}')
        img_paths = sorted(glob(self.glob_test_images))
        # Initialize data containers
        ids = []
        x = []
        for i, img_path in enumerate(tqdm(img_paths)):
            # Get the image id
            img_id = self.get_img_id(img_path)

            # Add the id
            ids.append(img_id)
            x.append(img_path)

        # Turn into numpy arrays
        x = np.array(x)
        print('X shape:', x.shape)
        if self.test_np:
            print('Using in memory image test dataset')
            return ImageNPRLEDataset(x, y=None, ids=np.array(ids))
        return ImageRLEDataset(x, y=None, ids=np.array(ids))

    @staticmethod
    def get_stratification_categories(train_dataset, num_categories=5):
        logging.info(
            f'Constructing {num_categories} stratification categories.')
        n = len(train_dataset)

        def label_gen(dataset):
            for mask_rle in tqdm(dataset.y):
                mask = rle_decoding(mask_rle, img_size=ORIG_IMG_SIZE)
                yield mask

        coverage = np.array([np.sum(y) for y in label_gen(train_dataset)])
        bounds = np.linspace(0.0, coverage.max(), num=num_categories)

        categories = np.zeros(n, dtype='int64')
        for i in range(1, len(bounds)):
            categories[(coverage > bounds[i-1]) & (coverage <= bounds[i])] = i
        return categories

    @staticmethod
    def save_submission(save_name, preds, test_ids, cutoff=0.5):
        new_test_ids = []
        rles = []
        # Figure out if we have to resize
        resize_imgs = preds.shape[1:3] != ORIG_IMG_SIZE
        logging.info('Resize the images: {}'.format(resize_imgs))
        for n, id_ in enumerate(tqdm(test_ids)):
            pred = preds[n]
            if resize_imgs:
                pred = resize(preds[n], ORIG_IMG_SIZE,
                              mode='constant', preserve_range=True)
            pred = pred >= cutoff
            if np.count_nonzero(pred) < EMPTY_THRESHOLD:
                rle = []
            else:
                rle = rle_encoding(pred)
            rles.append(rle)
            new_test_ids.append(id_)

        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(
            lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(save_name, index=False)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        # Figure out if we need to start a new run length
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_decoding(run_lengths, img_size=ORIG_IMG_SIZE):
    flat_size = ORIG_IMG_SIZE[0] * ORIG_IMG_SIZE[1]
    mask = np.zeros(flat_size)
    for i in range(0, len(run_lengths), 2):
        run_start = run_lengths[i]
        length = run_lengths[i + 1]
        mask[run_start:run_start + length] = 1.
    # Reshape the mask
    mask = mask.reshape(ORIG_IMG_SIZE).T[:, :, None]
    return mask


if __name__ == '__main__':
    import plot_utils as plot
    logging.getLogger().setLevel(logging.INFO)
    # Some basic tests
    road = RoadData()

    # Train data tests
    logging.info("Testing the train data")
    train = road.load_train()
    # print(f'X: {train.x}')
    # print(f'Y: {train.y}')
    # Try plotting some training images
    traingen = train.flow(batch_size=4, shuffle=True)
    images, masks = next(traingen)
    print(f'Image pixel range: ({np.min(images)}, {np.max(images)})')
    print(f'Mask pixel range: ({np.min(masks)}, {np.max(masks)})')
    plot.plot_img_masks(images, masks)
    # Test rle decoding
    logging.info("Test the rle decoding")
    dec_masks = np.array([rle_decoding(rle_encoding(mask)) for mask in masks])
    plot.plot_img_masks(masks, dec_masks)

    # Stratification tests
    logging.info("Test the stratification")
    categories = road.get_stratification_categories(train)
    print(f'Category Counts: {np.bincount(categories)}')

    # Test data tests
    logging.info("Test the test data")
    test = road.load_test()
    # print(f'X: {test.x}')
    # Try plotting some test images
    testgen = test.flow(batch_size=9, shuffle=True)
    plot.plot_img_grid(next(testgen))

    # Test the cropper
    logging.info("Test the cropper")
    from augmenters import Cropper
    traingen = train.flow(batch_size=16, shuffle=True)
    traingen = Cropper(crop_size=(128, 128), augment_labels=True)(traingen)
    images, masks = next(traingen)
    plot.plot_img_masks(images, masks)
