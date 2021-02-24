import tensorflow as tf
import numpy as np
import os, time, collections, shutil

class BaseNet(object):

    def __init__(self):
        self.regularizers = []

    def predict(self, data, labels=None, sess=None):
        """
        Predicts labels on the given data, labels are given, the loss on will be returned along with the predictions..

        :param data: The data for which the prediction should be performed
        :param labels: (OPTIONAL) The true labels, for which the loss should be computed
        :param sess: The tensorflow session for the model

        :returns: Either the predictions, or the predictions with the loss if true labels were given.
        """
        loss = 0
        size = data.shape[0]
        predictions = [0]*size
        sess = self._get_session(sess)

        # batch prediction
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            # last batch might be smaller than actual batch size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                # converting sparse matrices
                tmp_data = tmp_data.toarray()
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout:1}

            if labels is not None:
                batch_labels = np.zeros((self.batch_size, labels.shape[1], labels.shape[2]))
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]

        predictions = np.array(predictions)
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions


    def evaluate(self, data, labels, sess=None):
        """
        Evaluates the model on the given data.

        :param data: The data for which the labels should be predicted
        :param labels: The true labels for the data
        :param sess: The tensorflow session for the model

        :returns: The loss given the prediction and the true labels
        """
        predictions, loss = self.predict(data, labels, sess)

        return loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        """
        Fits model to the given training data, validates on given validation data.

        :param train_data: The training data
        :param train_labels: The true training labels
        :param val_data: The validation data
        :param val_labels: The true validation labels
        """
        print("fitting")
        sess = 

    def _get_session(self, sess):
        print("Getting session")
