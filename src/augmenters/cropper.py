import numpy as np
from pyjet.augmenters import Augmenter


class Cropper(Augmenter):

    def __init__(self, crop_size, labels=True, augment_labels=False):
        super(Cropper, self).__init__(
            labels=labels, augment_labels=augment_labels)
        self.crop_size = crop_size
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]

    def augment(self, x):
        batch_size, h, w = x.shape[:3]
        i = np.random.randint(h - self.crop_height, size=batch_size)
        j = np.random.randint(w - self.crop_width, size=batch_size)
        return np.array([x[ind, i[ind]:i[ind] + self.crop_height, j[ind]:j[ind] + self.crop_width] for ind in range(batch_size)])
