import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def chebyshev(L, X, K):
    """
    Returns Chebyshev polynomials T_k of order up to K.

    :param L:
    :param X:
    :param K:
    """
    M,N = X.shape

    assert L.dtype == X.dtype

    Xt = np.empty((K, M, N), L.dtype)

    Xt[0, ...] = X

    if K > 1:
        Xt[1, ...] = L.dot(X)

    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

def compute_loss(outputs, labels, loss, regularization, regularizers):
    if loss == "l1":
        data_loss = keras.losses.mean_absolute_error(y_true=labels, y_pred=outputs)
    else:
        data_loss = keras.losses.mean_squared_error(y_true=labels, y_pred=outputs)

    regularization *= tf.add_n(regularizers)

    loss = data_loss + regularization

    # loss averages

    averages = tf.train.ExponentialMovingAverage(0.9)
    op_averages = averages.apply([data_loss, regularization, loss])
    loss_average = tf.identity(averages.average(loss), name='control')
    return loss, loss_average
