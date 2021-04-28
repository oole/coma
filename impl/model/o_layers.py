import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from util import graph_util
import scipy.sparse

class cheb_filter(layers.Layer):
    """
    Chebyshev Convolutional layer

    :param channels: Size of each input sample
    :param K: Chebyshev filter size, i.5e. number of hops.
    :param use_bias: If set to false the layer will not learn an additive bias, the default is :obj: `True`.
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """
    def __init__(self, input_shape, laplacian, num_conv_filters, polynomial_order, **kwargs):
        super(cheb_filter, self).__init__(**kwargs)


        pass

    def call(self, input_tensor, training=False):
        pass


class bias_relu(layers.Layer):
    def __init__(self):
        super(bias_relu, self).__init__()
        self.b = tf.Variable(shape=(1, 1, 0), dtype=tf.float32)
        # what about regularization

    def call(self, input_tensor):
        return tf.nn.relu(input_tensor + self.b)