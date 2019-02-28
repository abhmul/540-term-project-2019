from math import sqrt, ceil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

w = 10
h = 10


def get_grid_shape(n):
    h = int(sqrt(n))
    w = int(ceil(n / h))
    return w, h


def plot_img_grid(images):
    n = len(images)
    w, h = get_grid_shape(n)
    fig = plt.figure(figsize=(8, 8))
    for i in range(n):
        # Show image
        ax = fig.add_subplot(h, w, i+1)
        ax.axis('off')
        ax.set_aspect('equal')
        # Fix the images if it has only 1 channel
        if images[i].ndim == 3 and images[i].shape[-1] == 1:
            images[i] = images[i].squeeze(axis=-1)

        cmap = 'gray' if images[i].ndim == 2 else None
        plt.imshow(images[i].astype(float), cmap=cmap)

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.show()


def plot_img_masks(images, masks):
    n = len(images)
    assert n == len(masks)
    image_masks = [img for img_mask in zip(images, masks) for img in img_mask]
    plot_img_grid(image_masks)
