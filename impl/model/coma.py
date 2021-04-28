import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import o_layers


class ComaModel(keras.Model):
    def __init__(self, laplacians, downsampling_matrices, upsampling_matrices, F, K, p, nz, nv, which_loss, F_0=1,
                 filter='chebyshev5', brelu='b1relu', pool='mpool1', unpool='poolwT',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9, regularization=0,
                 dropout=0, batch_size=100, eval_frequency=200, dir_name='', **kwargs):
        super(ComaModel, self).__init__(**kwargs)
        # Define
        self.laplacians = laplacians
        self.downsampling_matrices = downsampling_matrices
        self.upsampling_matrices = upsampling_matrices
        self.num_conv_filters = F
        self.num_conv_layers = len(F)
        self.polynomial_orders = K
        self.pooling_size = p
        self.size_latent = nz
        self.num_vertices = nv
        self.input_dimension = F_0

        self.which_loss = which_loss
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.momentum = momentum
        self.dir_name = dir_name

        input_mesh_size = laplacians[0].shape[0]
        print("building model")
        self.Encoder(input_mesh_size, self.input_dimension, self.num_conv_layers, self.laplacians,
                     self.num_conv_filters, self.polynomial_orders)
        # encoder:
        ## 4x
        ### Cheb + biased relu
        ### downsampling
        ## 1x
        ### Dense 20x32 -> 8

        # decoder
        ## 1x
        ### Dense 8 -> 20x32
        ## 4x
        ### Upsampling
        ### convolution + biased relu

        ## apply weight regularization.

    def call(self):
        pass


class Encoder(keras.Model):
    def __init__(self, input_size, input_dimension, num_conv_layers, laplacians, num_conv_filters, polynomial_orders,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input = layers.Input((None, input_size, input_dimension))
        self.cheb_blocks = []
        for i in range(num_conv_layers):
            self.cheb_blocks.append(
                ChebBlock(self.input.shape(), laplacians[i], num_conv_filters[i], polynomial_orders[i]))
        # encoder needs:
        # input shape (batch_size, input_size, F_0),

        pass

    def call(self):
        pass


class ChebBlock(layers.Layer):
    def __init__(self, input_shape, laplacian, num_conv_filters, polynomial_order):
        super(ChebBlock, self).__init__()
        self.filter = o_layers.cheb_filter(input_shape, laplacian, num_conv_filters, polynomial_order)
        self.bias_relu = None
        self.pool = None

    def call(self, input_tensor, training=False):
        x = self.filter(input_tensor)
        x = self.bias_relu(x)
        x = self.pool(x)
        return x


class Decoder(keras.Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        pass

    def call(self):
        pass
