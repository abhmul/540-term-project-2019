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


def encoder(x, filters=32, n_block=3, kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2, no_pool=False):
    skip = []
    for i in range(n_block):

        # Add the dropout if necessary
        x = SpatialDropout2D(dropout)(x) if dropout else x

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

        skip.append(x)
        if no_pool:
            x = Conv2D(filters * 2**i, kernel_size=(2, 2), strides=(2, 2))(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = Activation(activation)(x)
        else:
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def dilated_bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
                       kernel_size=(3, 3), activation='relu', batchnorm=True,
                       dropout=0.2):
    dilated_layers = []

    # Add the dropout if necessary
    x = SpatialDropout2D(dropout)(x) if dropout else x

    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       padding='same', dilation_rate=2**i)(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = Activation(activation)(x)

            dilated_layers.append(x)

    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            px = x
            px = Conv2D(filters_bottleneck, kernel_size,
                        padding='same', dilation_rate=2**i)(px)
            px = batchnorm_if_true(batchnorm)(px)
            px = Activation(activation)(px)

            dilated_layers.append(px)

    return add(dilated_layers)


def bottleneck(x, filters_bottleneck, depth=3,
               kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2):

    # Add the dropout if necessary
    x = SpatialDropout2D(dropout)(x) if dropout else x

    for i in range(depth):
        x = Conv2D(filters_bottleneck, kernel_size,
                   padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

    return x


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu', batchnorm=True, dropout=0.2, use_deconv=False):
    for i in reversed(range(n_block)):
        if use_deconv:
            x = Conv2DTranspose(
                filters * 2 ** i, kernel_size=(2, 2), strides=(2, 2))(x)
            x = batchnorm_if_true(batchnorm)(x)
            x = Activation(activation)(x)
        else:
            x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

        x = concatenate([skip[i], x])

        # Add the dropout if necessary
        x = SpatialDropout2D(dropout)(x) if dropout else x

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

        x = Conv2D(filters * 2**i, kernel_size, padding='same')(x)
        x = batchnorm_if_true(batchnorm)(x)
        x = Activation(activation)(x)

    return x


def get_simple_unet(
    input_shape=(512, 512, 3),
    filters=32,
    n_block=3,
    bottleneck_depth=3,
    dropout=0.2,
    optimizer=None,
    loss=bce_dice_loss,
    use_deconv=False,
    no_pool=False
):
    return get_unet(
        input_shape=input_shape,
        filters=filters,
        n_block=n_block,
        dropout=dropout,
        optimizer=optimizer,
        loss=bce_dice_loss,
        n_class=1,
        use_deconv=use_deconv,
        no_pool=no_pool,
        dilated=False,
        bottleneck_depth=bottleneck_depth,
        mode=None
    )


def get_unet(
    input_shape=(512, 512, 3),
    filters=44,
    n_block=3,
    dropout=0.2,
    optimizer=None,
    loss=bce_dice_loss,
    n_class=1,
    use_deconv=False,
    no_pool=False,
    dilated=False,
    bottleneck_depth=3,
    mode='cascade'
):

    # Default to an RMSProp optimizer with lr=0.0001
    if optimizer is None:
        optimizer = RMSprop(0.0001)
    if isinstance(optimizer, dict):
        optimizer = build_optimizer(optimizer)

    inputs = Input(input_shape)

    enc, skip = encoder(inputs, filters, n_block,
                        no_pool=no_pool, dropout=dropout)
    if dilated:
        bottle = dilated_bottleneck(enc,
                                    filters_bottleneck=filters * 2 ** n_block,
                                    depth=bottleneck_depth,
                                    dropout=dropout,
                                    mode=mode)
    else:
        bottle = bottleneck(enc,
                            filters_bottleneck=filters * 2 ** n_block,
                            depth=bottleneck_depth,
                            dropout=dropout)
    dec = decoder(bottle, skip, filters, n_block,
                  use_deconv=use_deconv, dropout=dropout)

    classify = Conv2D(
        n_class, (1, 1), activation='sigmoid' if n_class == 1 else 'softmax')(dec)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[dice_coef])

    return model


def get_dilated_unet(
        input_shape=(512, 512, 3),
        mode='cascade',
        filters=32,
        n_block=3,
        bottleneck_depth=6,
        dropout=0.2,
        optimizer=None,
        loss=bce_dice_loss,
        n_class=1,
        use_deconv=False,
        no_pool=False
):

    return get_unet(
        input_shape=input_shape,
        filters=filters,
        n_block=n_block,
        dropout=dropout,
        optimizer=optimizer,
        loss=bce_dice_loss,
        n_class=1,
        use_deconv=use_deconv,
        no_pool=no_pool,
        dilated=True,
        bottleneck_depth=bottleneck_depth,
        mode=mode
    )
