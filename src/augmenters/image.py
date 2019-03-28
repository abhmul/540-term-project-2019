import numpy as np
from pyjet.augmenters import Augmenter


class ImageAugmenter(Augmenter):

    def __init__(self, image_augmenter, labels=True, augment_labels=False):
        super(ImageAugmenter, self).__init__(
            labels=labels, augment_labels=augment_labels)
        self.image_augmenter = image_augmenter

    def _augment(self, batch):
        # Split the batch if necessary
        if self.labels:
            x, y = batch
            if self.augment_labels:
                x, y = self.augment(x, y)
            else:
                x = self.augment(x)
            return x, y

        else:
            x = batch
            return self.augment(x)

    def augment(self, *tensors):
        aug = self.image_augmenter.to_deterministic()
        return [aug.augment_images(t) for t in tensors]
