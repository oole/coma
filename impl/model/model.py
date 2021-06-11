import tensorflow as tf
import numpy as np
import os, time, collections, shutil
from model.o_layers import encoder_block, decoder_block
from tensorflow import keras


class coma_ae(keras.Model):
    """
    Convolutional Mesh Autoencoder Model.

    :param num_input_features: The number of output features of the last decoder block.
    :param num_features: A list of number of features for the decoding blocks
    :param laplacians: A list of laplacians for the encoder and decoder
    :param downsampling_transformations: A list of downsampling transformations to be applied by the encoder
    :param upsampling_transformations: A list of upsampling transformations to be applied by the decoder
    :param Ks: A list of polynomial orders to be applied by the decoding blocks
    :param num_latent: The size of the latent representation of the meshes
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.Model`.
    """

    def __init__(self,
                 num_input_features,
                 num_features,
                 laplacians,
                 downsampling_transformations,
                 upsampling_transformations,
                 Ks,
                 num_latent,
                 batch_size,
                 **kwargs):
        super(coma_ae, self).__init__(**kwargs)
        self.encoder = encoder(num_input_features=num_input_features,
                               num_features=num_features,
                               laplacians=laplacians,
                               downsampling_transformations=downsampling_transformations,
                               Ks=Ks,
                               num_latent=num_latent,
                               batch_size=batch_size)
        self.decoder = decoder(num_output_features=num_input_features,
                               num_features=num_features,
                               laplacians=laplacians,
                               upsampling_transformations=upsampling_transformations,
                               Ks=Ks,
                               batch_size=batch_size)

    def call(self, input_tensor):
        x = self.encoder(input_tensor)
        x = self.decoder(x)
        return x

    def encode(self, input_tensor):
        x = self.encoder(input_tensor)
        return x

    def decode(self, input_tensor):
        x = self.decoder(input_tensor)
        return x

    def model(self, input_shape, batch_size):
        x = keras.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class encoder(keras.Model):
    """
    Encoder model consisting of an encoder blocks.

    Concatenates a number of decoder blocks, depending on the length of the given list of num_features.

    :param num_input_features: The number of input features for the first encoder block
    :param num_features: A list of number of features for the decoding blocks
    :param laplacians: A list of laplacians for the decoding blcks
    :param downsampling_transformations: A list of downsampling transformations to be applied by the encoding blocks
    :param Ks: A list of polynomial orders to be applied by the decoding blocks
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.Model`.
    """

    def __init__(self,
                 num_input_features,
                 num_features,
                 laplacians,
                 downsampling_transformations,
                 Ks,
                 num_latent,
                 batch_size,
                 **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.encoder_blocks = []
        for i in range(len(num_features)):
            if i == 0:
                self.encoder_blocks.append(encoder_block(laplacian=laplacians[i],
                                                         K=Ks[i],
                                                         input_features=num_input_features,
                                                         output_features=num_features[i],
                                                         downsampling_transformation=downsampling_transformations[i],
                                                         batch_size=batch_size))
            else:
                self.encoder_blocks.append(encoder_block(laplacian=laplacians[i],
                                                         K=Ks[i],
                                                         input_features=num_features[i - 1],
                                                         output_features=num_features[i],
                                                         downsampling_transformation=downsampling_transformations[i],
                                                         batch_size=batch_size))

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_latent,
                                           activation=keras.activations.relu,
                                           use_bias=True,
                                           bias_regularizer=keras.regularizers.L2(),
                                           bias_initializer=keras.initializers.constant(value=0.1),
                                           kernel_regularizer=keras.regularizers.L2(),
                                           kernel_initializer=keras.initializers.truncated_normal(mean=0.0, stddev=0.1))

    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def model(self, input_shape, batch_size):
        """
        Helper to enable model summary encoder.model(input_shape).summary()
        """
        x = keras.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class decoder(keras.Model):
    """
    Decoder Model consisting of decoder blocks.

    Concatenates a number of decoder blocks, depending on the length of the given list of num_features.

    :param num_output_features: The number of output features of the last decoder block.
    :param num_features: A list of number of features for the decoding blocks
    :param laplacians: A list of laplacians for the decoding blcks
    :param upsampling_transformations: A list of upsampling transformations to be applied by the decoding blocks
    :param Ks: A list of polynomial orders to be applied by the decoding blocks
    :param batch_size: The batch size
    :param **kwargs: (optional) additional arguments of :class: `tf.keras.Model`.
    """

    def __init__(self,
                 num_output_features,
                 num_features,
                 laplacians,
                 upsampling_transformations,
                 Ks,
                 batch_size,
                 **kwargs):
        super(decoder, self).__init__(**kwargs)
        initial_size = upsampling_transformations[-1].shape[1]
        initial_num_features = num_features[-1]
        self.fc = keras.layers.Dense(initial_size * initial_num_features,
                                     activation=keras.activations.relu,
                                     use_bias=True,
                                     bias_regularizer=keras.regularizers.L2(),
                                     bias_initializer=keras.initializers.constant(value=0.1),
                                     kernel_regularizer=keras.regularizers.L2(),
                                     kernel_initializer=keras.initializers.truncated_normal(mean=0.0, stddev=0.1))

        self.reshape = keras.layers.Reshape((initial_size, initial_num_features))
        self.decoder_blocks = []
        for i in range(len(num_features)):
            if i == (len(num_features) - 1):
                # Last layer has given output feature size
                self.decoder_blocks.append(decoder_block(laplacian=laplacians[-i - 1],
                                                         K=Ks[-i - 1],
                                                         input_features=num_features[-i - 1],
                                                         output_features=num_output_features,
                                                         upsampling_transformation=upsampling_transformations[-i - 1],
                                                         batch_size=batch_size))
            else:
                self.decoder_blocks.append(decoder_block(laplacian=laplacians[-i - 1],
                                                         K=Ks[-i - 1],
                                                         input_features=num_features[-i - 1],
                                                         output_features=num_features[-i - 2],
                                                         upsampling_transformation=upsampling_transformations[-i - 1],
                                                         batch_size=batch_size))

    def call(self, input_tensor):
        x = self.fc(input_tensor)
        x = self.reshape(x)
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
        return x

    def model(self, input_shape, batch_size):
        """
            Helper to enable model summary decoder.model(input_shape).summary()
        """
        x = keras.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
