import numpy as np
import tensorflow as tf
import keras.backend as K


def castF(x):
    return K.cast(x, K.floatx())


def castB(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def iou(true, pred, pred_t=0.5):
    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, pred_t))

    return iou_loss_core(true, pred)


def mean_iou(true, pred, pred_t=0.5):  # any shape can go - can't be a loss function

    thresholds = [0.5 + (i*.05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, pred_t))

    # total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    # to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    # separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(testTrue, testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in thresholds]

    # mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    # to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


def mean_iou_t(true, pred):
    thresholds = np.linspace(0.05, 0.95, num=19)
    return K.max(K.stack([mean_iou(true, pred, pred_t=t) for t in thresholds]))
