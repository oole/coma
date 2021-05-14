import tensorflow as tf
from tensorflow.keras import layers
from util import graph_util
from scipy import sparse
import numpy as np

class cheb_conv(layers.Layer):
    """
    Chebyshev Convolutional layer

    -> Convolutional Neural Networks on Graphs with Fast Localized Spectral fFiltering

    :param input_features: Size of each input sample
    :param output_features: The number of output features
    :param K: Chebyshev filter size
    :param laplacian: The laplacian for the input mesh
    :param batch_size: The batch_size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """

    def __init__(self, K, input_features, output_features, laplacian, batch_size, **kwargs):
        super(cheb_conv, self).__init__(**kwargs)
        self.K = K
        self.input_features = input_features
        self.output_features = output_features
        self.laplacian = laplacian
        self.batch_size = batch_size

    def build(self, input_shape):
        # build layer weights
        L = sparse.csr_matrix(self.laplacian)
        L = graph_util.rescale_laplacian(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse.reorder(L)
        self.L = L

        self.w = self.add_weight(
            name='w',
            shape=(self.input_features * self.K, self.output_features),
            initializer=tf.keras.initializers.truncated_normal(),
            trainable=True,
            regularizer=tf.keras.regularizers.L2()
        )

    def call(self, input_tensor):
        # tansform input to chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])
        x0 = tf.reshape(x0, [tf.shape(input_tensor)[1], self.input_features * self.batch_size])
        x = tf.expand_dims(x0, 0)
        if self.K > 1:
            x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
            x = tf.concat([x, tf.expand_dims(x1, 0)], axis=0)

        for k in range(2, self.K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.L, x1) - x0
            x = tf.concat([x, tf.expand_dims(x2, 0)], axis=0)
            x0, x1 = x1, x2

        x = tf.reshape(x, [self.K, tf.shape(input_tensor)[1], self.input_features, self.batch_size])

        x = tf.transpose(x, perm=[3, 1, 2, 0])

        x = tf.reshape(x, [self.batch_size * tf.shape(input_tensor)[1], self.input_features * self.K])
        # compute conv
        x = tf.matmul(x, self.w)

        return tf.reshape(x, [self.batch_size, tf.shape(input_tensor)[1], self.output_features])


class sampling(layers.Layer):
    """
    Sampling layer.

    :param sampling_transformation: The pre-computed up- or downsampling transformation that should be applied
    :param input_features: The number of input features
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """
    def __init__(self, sampling_transformation, input_features, batch_size, **kwargs):
        super(sampling, self).__init__(**kwargs)
        self.sampling_transformation = sampling_transformation
        self.input_features = input_features
        self.batch_size = batch_size

    def build(self, input_shape):
        self.input_mesh_size = self.sampling_transformation.shape[1]
        self.output_mesh_size = self.sampling_transformation.shape[0]
        D = sparse.csr_matrix(self.sampling_transformation)
        D = D.tocoo()
        indices = np.column_stack((D.row, D.col))
        D = tf.SparseTensor(indices, D.data, D.shape)
        self.D = tf.sparse.reorder(D)

    def call(self, input_tensor):
        x = tf.transpose(input_tensor, perm=[1, 2, 0])
        x = tf.reshape(x, [self.input_mesh_size, self.input_features * self.batch_size])
        x = tf.sparse.sparse_dense_matmul(self.D, x)
        x = tf.reshape(x, [self.output_mesh_size, self.input_features, self.batch_size])
        x = tf.transpose(x, perm=[2, 0, 1])
        return x


class encoder_block(layers.Layer):
    """
    Encoder block consisting of a chebychev convolution layer followed by a downsampling layer

    :param laplacian: The laplacian for the chebyshev filter
    :param K: The polynomial order to be used by the chebyshev filter
    :param input_features: The number of input features
    :param output_features: The number of output features
    :param downsampling tansformation: The downsampling transformation to be applied
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """
    def __init__(self,
                 laplacian,
                 K,
                 input_features,
                 output_features,
                 downsampling_transformation,
                 batch_size, **kwargs):
        super(encoder_block, self).__init__(**kwargs)
        self.cheb_1 = cheb_conv(input_features=input_features,
                                output_features=output_features,
                                K=K,
                                laplacian=laplacian,
                                batch_size=batch_size)
        self.bias_relu_1 = bias_relu()
        self.downsampling_1 = sampling(sampling_transformation=downsampling_transformation,
                                       input_features=output_features,
                                       batch_size=batch_size)

    def call(self, input_tensor):
        x = self.cheb_1(input_tensor)
        x = self.bias_relu_1(x)
        x = self.downsampling_1(x)
        return x


class decoder_block(layers.Layer):
    """
    Decoder block consisting of an upsampling layer followed by a chebyshev convolution.

    :param laplacian: The laplacian for the chebyshev filter
    :param K: The polynomial order to be used by the chebyshev filter
    :param input_features: The number of input features
    :param output_features: The number of output features
    :param downsampling tansformation: The upsampling transformation to be applied
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.layers.Layer`.
    """
    def __init__(self, laplacian, K, input_features, output_features, upsampling_transformation, batch_size, **kwargs):
        super(decoder_block, self).__init__(**kwargs)
        self.upsampling_1 = sampling(sampling_transformation=upsampling_transformation,
                                     input_features=input_features,
                                     batch_size=batch_size)

        self.dec_cheb_1 = cheb_conv(
            input_features=input_features,
            output_features=output_features,
            K=K,
            laplacian=laplacian,
            batch_size=batch_size)
        self.bias_relu_1 = bias_relu()

    def call(self, input_tensor):
        x = self.upsampling_1(input_tensor)
        x = self.dec_cheb_1(x)
        x = self.bias_relu_1(x)
        return x


class bias_relu(layers.Layer):
    """
    Custom relu layer that adds a bias.
    """
    def __init__(self, **kwargs):
        super(bias_relu, self).__init__(**kwargs)

    def build(self, input_shape):
        _, mesh_size, num_features = input_shape
        self.b = self.add_weight(
            name='bias',
            shape=(1, mesh_size, num_features),
            initializer=tf.keras.initializers.constant(value=0.1, ),
            trainable=True,
            regularizer=tf.keras.regularizers.L2()
        )

    def call(self, input_tensor):
        return tf.nn.relu(input_tensor + self.b)


