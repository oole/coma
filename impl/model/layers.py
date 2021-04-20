import tensorflow as tf
from tensorflow import keras

class ChebyConv(tf.keras.layers.Layer):
    """
    Chebyshev Convolutional layer

    :param channels: Size of each input sample
    :param K: Chebyshev filter size, i.e. number of hops.
    :param use_bias: If set to false the layer will not learn an additive bias, the default is :obj: `True`.
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """
    def __init__(self, channels, K=1, activation=None, use_bias=True, **kwargs):
        super(ChebyConv, self).__init__(**kwargs)

