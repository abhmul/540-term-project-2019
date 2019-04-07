from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, SpatialDropout2D, Conv2DTranspose
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras import optimizers as keras_optimizers
from keras.optimizers import RMSprop
from kaggleutils import dump_args

from .losses_keras import (
    binary_crossentropy,
    dice_loss,
    bce_dice_loss,
    weighted_bce_dice_loss,
    bce_iou_loss,
    dice_coef
)
from .metrics_keras import (
    mean_iou,
    mean_iou_t,
    iou
)


def build_optimizer(params):
    optim_name = params.pop('name')
    optimizer = getattr(keras_optimizers, optim_name)(**params)
    params['name'] = optim_name
    return optimizer


def batchnorm_if_true(batchnorm):
    return BatchNormalization() if batchnorm else lambda x: x


def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2, no_pool=False):
    skip = []
    activation_layer = Activation(activation)
    for i in range(n_block):

        # Add the dropout if necessary
        x = SpatialDropout2D(dropout)(x) if dropout else x

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

        skip.append(x)
        if no_pool:
            x = Conv2D(filters * 2**i, kernel_size=(2, 2), stride=(2, 2))(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = activation_layer(x)
        else:
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def dilated_bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
                       kernel_size=(3, 3), activation='relu', batchnorm=True,
                       dropout=0.2):
    dilated_layers = []
    activation_layer = Activation(activation)

    # Add the dropout if necessary
    x = SpatialDropout2D(dropout)(x) if dropout else x

    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       padding='same', dilation_rate=2**i)(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = activation_layer(x)

            dilated_layers.append(x)

    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            px = x
            px = Conv2D(filters_bottleneck, kernel_size,
                        padding='same', dilation_rate=2**i)(px)
            px = batchnorm_if_true(batchnorm)(px)
            px = activation_layer(px)

            dilated_layers.append(px)

    return add(dilated_layers)


def bottleneck(x, filters_bottleneck, depth=3,
               kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2):
    activation_layer = Activation(activation)

    # Add the dropout if necessary
    x = SpatialDropout2D(dropout)(x) if dropout else x

    for i in range(depth):
        x = Conv2D(filters_bottleneck, kernel_size,
                   padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

    return x


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2, use_deconv=False):
    activation_layer = Activation(activation)
    for i in reversed(range(n_block)):
        if use_deconv:
            x = Conv2DTranspose(
                filters * 2 ** i, kernel_size=(2, 2), strides=(2, 2))(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = activation_layer(x)
        else:
            x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

        x = concatenate([skip[i], x])

        # Add the dropout if necessary
        x = SpatialDropout2D(dropout)(x) if dropout else x

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = activation_layer(x)

    return x


def get_simple_unet(
    input_shape=(512, 512, 3),
    filters=4,
    n_block=3,
    bottleneck_depth=3,
    optimizer=None,
    loss=bce_dice_loss,
    n_class=1,
    use_deconv=False,
    no_pool=False
):
    # Default to an Adam optimizer
    if optimizer is None:
        optimizer = 'adam'
    if isinstance(optimizer, dict):
        optimizer = build_optimizer(optimizer)

    inputs = Input(input_shape)

    enc, skip = encoder(inputs, filters, n_block, no_pool=no_pool)
    bottle = bottleneck(enc, filters_bottleneck=filters *
                        2**n_block, depth=bottleneck_depth)
    dec = decoder(bottle, skip, filters, n_block, use_deconv=use_deconv)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[dice_coef])

    return model


def get_dilated_unet(
        input_shape=(None, None, 3),
        mode='cascade',
        filters=44,
        n_block=3,
        optimizer=None,
        loss=bce_dice_loss,
        n_class=1,
        use_deconv=False,
        no_pool=False
):

    # Default to an RMSProp optimizer with lr=0.0001
    if optimizer is None:
        optimizer = RMSprop(0.0001)
    if isinstance(optimizer, dict):
        optimizer = build_optimizer(optimizer)

    inputs = Input(input_shape)

    enc, skip = encoder(inputs, filters, n_block, no_pool=no_pool)
    bottle = dilated_bottleneck(enc, filters_bottleneck=filters *
                                2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block, use_deconv=use_deconv)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[dice_coef])

    return model
