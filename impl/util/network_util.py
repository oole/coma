import os
import tensorflow as tf
import datetime


class model_util(object):

    def get_checkpoint_callback(self, path: str):
        """
        Adds a checkpoint callback that stores the weights to the given path
        :param path:
        :return:
        """
        checkpoint_dir = os.path.dirname(path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                         save_weights_only=True,
                                                         verbose=1)
        return cp_callback

    def get_tensorboard_callback(selfs, logdir: str):

        date_logdir = logdir + "/" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=date_logdir, histogram_freq=1)

        return tensorboard_callback

    def save_model_weights(self, path: str, model: tf.keras.Model):
        """
        Save the weights to the given path
        :param path:
        :param model:
        :return:
        """
        model.save_weights(path)

    def load_model_weights(self, path: str, model: tf.keras.Model):
        """
        Load the weights from the given path
        :param path:
        :param str:
        :param model:
        :return:
        """
        model.load_weights(path)
        return model

    def save_model(self, path: str, model: tf.keras.Model):
        """
        Saves complete model as h5py

        :param path: The path where the model should be stored
        :param model: The model that should be stored
        """
        model.save(path)
