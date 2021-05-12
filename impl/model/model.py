import tensorflow as tf
import numpy as np
import os, time, collections, shutil
from o_layers import encoder_block, decoder_block
from tensorflow import keras


class coma_ae(keras.Model):
    def __init__(self,
                 num_features,
                 laplacians,
                 downsampling_transformations,
                 upsampling_transformations,
                 Ks,
                 num_latent,
                 **kwargs):
        super(coma_ae, self).__init__(**kwargs)
        self.encoder = encoder(num_features=num_features,
                               laplacians=laplacians,
                               downsampling_transformations=downsampling_transformations,
                               Ks=Ks,
                               num_latent=num_latent)
        self.decoder = decoder(num_features=num_features,
                               laplacians=laplacians,
                               upsampling_transformations=upsampling_transformations,
                               Ks=Ks)

    def call(self, input_tensor):
        x = self.encoder(input_tensor)
        x = self.decoder(x)
        return x

    def model(self, input_shape):
        x = keras.Input(shape=(input_shape[1], input_shape[2]), batch_size=input_shape[0])
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class encoder(keras.Model):
    """
    Model for the encoder
    """
    def __init__(self, num_features, laplacians, downsampling_transformations, Ks, num_latent, **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.encoder_blocks = []
        for i in range(len(num_features)):
            self.encoder_blocks.append(encoder_block(laplacian=laplacians[i],
                                                     K=Ks[i],
                                                     input_features=num_features[i],
                                                     output_features=num_features[i],
                                                     downsampling_transformation=downsampling_transformations[i]))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_latent)

    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def model(self, input_shape):
        """
        Helper to enable model summary encoder.model(input_shape).summary()
        """
        x = keras.Input(shape=(input_shape[1],input_shape[2]), batch_size=input_shape[0])
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class decoder(keras.Model):
    def __init__(self, num_features, laplacians, upsampling_transformations, Ks, **kwargs):
        super(decoder, self).__init__(**kwargs)
        initial_size = upsampling_transformations[-1].shape[1]
        initial_num_features = num_features[-1]
        self.fc = keras.layers.Dense(initial_size * initial_num_features)
        self.reshape = keras.layers.Reshape((initial_size, initial_num_features))
        self.decoder_blocks = []
        for i in range(len(num_features)):
            self.decoder_blocks.append(decoder_block(laplacian=laplacians[-i - 1],
                                                     K=Ks[-i - 1],
                                                     input_features=num_features[-i - 1],
                                                     output_features=num_features[-i - 1],
                                                     upsampling_transformation=upsampling_transformations[-i - 1]))

    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
        return x

    def model(self, input_shape):
        """
            Helper to enable model summary decoder.model(input_shape).summary()
        """
        x = keras.Input(shape=(input_shape[1], input_shape[2]), batch_size=input_shape[0])
        print(x.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
