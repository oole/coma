import numpy as np
from psbody.mesh import Mesh


class MeshData(object):
    def __init__(self, number_val: int, train_file: str, test_file: str, reference_mesh_file: str):
        """
        Constructor; create a new Mesh Dataset, consisting of train, validation and test set.

        :param number_val: The number of validation samples from the training dataset
        :param train_file: The file containing the training samples
        :param test_file: The file containing the test samples
        :param reference_mesh_file: The path to the reference mesh file
        """
        self.number_val = number_val
        self.train_file = train_file
        self.test_file = test_file
        self.vertices_train = None
        self.vertices_val = None
        self.vertices_test = None
        self.N = None
        self.n_vertex = None
        self.mean = None
        self.std = None

        self.load()
        self.reference_mesh = Mesh(filename=reference_mesh_file)
        self.normalize()

    def load(self):
        """
        Load the mesh data from the train_file and test_file. The train_file will be split into training and validation
        by splitting the train_file, according to the numer of validation samples.
        """
        vertices_train = np.load(self.train_file)
        self.mean = np.mean(vertices_train, axis=0)
        self.std = np.std(vertices_train, axis=0)

        #split in to train val
        self.vertices_train = vertices_train[:-self.number_val]
        self.vertices_val = vertices_train[-self.number_val:]

        self.n_vertex = self.vertices_train.shape[0]

        self.vertices_test = np.load(self.test_file)

    def normalize(self):
        """
        Normalizes the dataset by substracting the training set mean and dividing by the training
        set standard deviation.
        """
        # train
        self.vertices_train = self.vertices_train - self.mean
        self.vertices_train = self.vertices_train/self.std

        self.vertices_val = self.vertices_val - self.mean
        self.vertices_val = self.vertices_val/self.std

        self.vertices_test = self.vertices_test - self.mean
        self.vertices_test = self.vertices_test/self.std

        self.N = self.vertices_train.shape[0]
