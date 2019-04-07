import numpy as np
from pyjet.augmenters import Augmenter


class ImageAugmenter(Augmenter):

    def __init__(self, image_augmenter, labels=True, augment_labels=False, store_history=False):
        super(ImageAugmenter, self).__init__(
            labels=labels, augment_labels=augment_labels)
        self.image_augmenter = image_augmenter
        self.store_history = store_history
        self.past_augs = []

    def _augment(self, batch):
        # Split the batch if necessary
        if self.labels:
            x, y = batch
            if self.augment_labels:
                x, y = self.augment(x, y)
            else:
                x = self.augment(x)[0]
            return x, y

        else:
            x = batch
            return self.augment(x)[0]

    def augment(self, *tensors):
        aug = self.image_augmenter.to_deterministic()
        if self.store_history:
            self.past_augs.append(aug)
        return [aug.augment_images(t) for t in tensors]

    def clear_history(self):
        self.past_augs = []


class FlipAugmenter(Augmenter):

    def __init__(self, flipud=False, fliplr=False, labels=True, augment_labels=False):
        super(FlipAugmenter, self).__init__(
            labels=labels, augment_labels=augment_labels)
        self.flipud = flipud
        self.fliplr = fliplr

    def augment(self, x):
        if self.flipud:
            x = x[:, ::-1]
        if self.fliplr:
            x = x[:, :, ::-1]
        return x

    def reverse_augment(self, x):
        print(x.shape)
        if self.fliplr:
            x = x[:, :, ::-1]
        if self.flipud:
            x = x[:, ::-1]
        return x
