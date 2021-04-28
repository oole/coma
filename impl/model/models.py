import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import o_layers as olayers


class MeshAutoencoder(keras.Model):
    def __init__(self, L, D, U, F, K, p, nz, nv, which_loss, F_0 = 1, filter='chebyshev5', brelu='b1relu', pool='mpool1', unpool='poolwT',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9, regularization=0,
                 dropout=0, batch_size=100, eval_frequency=200, dir_name=''):
        """
        Convolutional Mesh Autoencoder Model.


        :param F: The number of features
        :param K: A list of polynomial orders, e.g. filter sizes
        :param p: Pooling size, 1, or power of 2 (reduction by factor 2 at each layer)

        :param M: The number of features per sample, the last layer index M[-1] is the number of classes

        :param filter: The filtering operation, that should be used
        :param brelu: Specifies bias and relu, e.g. b1relu or b2relu
        :param pool: Specifies pooling, e.g. mpool1

        :param num_epochs: The number of training epochs
        :param learning_rate: The initial learning rate
        :param decay_rate: The base of exponential decay (no decay = 1)
        :param decay_steps: The number of steps after which the learning rate should decay
        :param momentum: Specifies the momentum, 0 means no momentum

        :param regularization: L2 regularization of weights and biases
        :param dropout: Dropout layer's probability to keep connections, 1 means no dropout
        :param batch_size: The Batch size
        :param eval_frequency: The number of steps between evaluations of the model

        :param dir_name: Path to the directory for summary and model parameters
        """
        super(MeshAutoencoder, self).__init__()
        Parse parameters
        self.regularizers = []
        # Build the blocks of the model
        M_0 = L[0].shape[0]

        Ngconv = len(p)
        Nfc = len(nz)



        # This is where the model is built
        # Inputs

        # Model

        ## Encoder
        self.encoder = olayers.Encoder(self.L, self.F, self.K)
        ## Decoder

        # Initialization

        # Summary setup




    def call(self, input_tensor, training=False):3
        pass

    def model(self):
