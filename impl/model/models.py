import tensorflow as tf


class MeshAutoencoder(tf.keras.Model):
    def __init__(self, M, F, **kwargs):
        super(MeshAutoencoder, self).__init__(**kwargs)
        self._build(M, F)

    def _loss(self):

    def _accuracy(self):

    def _build(self, M, F):
        self.layers.append(())



    # to control fitting, override train_step(self, data)
    def train_step(self, data):
