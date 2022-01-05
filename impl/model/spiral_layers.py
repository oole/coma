import tensorflow as tf
from tensorflow.keras import layers
from scipy import sparse
import numpy as np


class spiral_conv(layers.Layer):
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

    def __init__(self, spirals, input_features, output_features, batch_size, regularization=None, **kwargs):
        super(spiral_conv, self).__init__(**kwargs)
        self.spirals = spirals
        self.spiral_length = spirals[0].size
        self.num_nodes = spirals.shape[0]
        self.spiral_indices = spirals.reshape(-1)
        self.input_features = input_features
        self.output_features = output_features
        self.batch_size = batch_size
        self.regularization = regularization

    def build(self, input_shape):
        # build layer weights
        if self.regularization is None:
            self.w = self.add_weight(name='w',
                            shape=(self.input_features * self.spiral_length, self.output_features),
                            initializer=tf.keras.initializers.glorot_uniform(),
                            trainable=True)

        else:
            print("Add L1 output-weight regularization")
            self.w = self.add_weight(name='w',
                                     shape=(self.input_features * self.spiral_length, self.output_features),
                                     initializer=tf.keras.initializers.glorot_uniform(),
                                     trainable=True)

    def call(self, input_tensor):
        #
        input_shape = tf.shape(input_tensor)
        print("spiral-conv - input shape: " + str(input_shape))
        print("input tensor:")
        print("dim 0: " + str(input_tensor[0]))
        print("dim 1: " + str(input_tensor[1]))
        print("dim 2: " + str(input_tensor[2]))
        print("Spirals shape: " + str(self.spirals.shape))
        print("spiral indices: " + str(self.spiral_indices))
        print("Input shape [0]: " + str(input_shape[0]))
        print("INPUT SHAPE 0: 2")
        x = tf.gather(input_tensor, self.spiral_indices, axis=1)
        print("x after gather: " + str(tf.shape(x)))
        print("x[0] after gather: " + str(x[0]))
        x = tf.reshape(x, [self.batch_size, self.num_nodes, -1])
        print("x shape: after reshape: " + str(tf.shape(x)))
        print("x after reshape:" + str(x))
        print("x[0] after reshape: " + str(x[0]))
        print("num_nodes: " + str(self.num_nodes))

        print("Vertex spirals reshaped to matrix for convolution.")
        print("Weights: " + str(self.w))
        x = tf.matmul(x, self.w)
        print("x after matmul: " + str(x))
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
                 spirals,
                 input_features,
                 output_features,
                 downsampling_transformation,
                 batch_size, **kwargs):
        super(encoder_block, self).__init__(**kwargs)
        self.spiral_1 = spiral_conv(input_features=input_features,
                                    output_features=output_features,
                                    spirals=spirals,
                                    batch_size=batch_size)
        self.bias_relu_1 = bias_relu()
        self.downsampling_1 = sampling(sampling_transformation=downsampling_transformation,
                                       input_features=output_features,
                                       batch_size=batch_size)

    def call(self, input_tensor):
        x = self.spiral_1(input_tensor)
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

    def __init__(self, spirals, input_features, output_features, upsampling_transformation, batch_size, **kwargs):
        super(decoder_block, self).__init__(**kwargs)
        self.upsampling_1 = sampling(sampling_transformation=upsampling_transformation,
                                     input_features=input_features,
                                     batch_size=batch_size)

        self.dec_spiral_1 = spiral_conv(input_features=input_features,
                                        output_features=output_features,
                                        spirals=spirals,
                                        batch_size=batch_size)

        self.bias_relu_1 = bias_relu()

    def call(self, input_tensor):
        x = self.upsampling_1(input_tensor)
        x = self.dec_spiral_1(x)
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
            shape=(1, 1, num_features),
            initializer=tf.keras.initializers.constant(value=0.1),
            trainable=True,
        )

    def call(self, input_tensor):
        return tf.nn.relu(input_tensor + self.b)
