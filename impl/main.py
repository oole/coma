import os
import copy
import time
import numpy as np
import argparse

from numpy import save

from util import mesh_sampling, mesh_util
from util.log_util import date_print
from psbody.mesh import MeshViewers, Mesh
from data import meshdata
import util.graph_util as graph
from model.model import coma_ae
from tensorflow import keras
import tensorflow as tf

## experimental config for memory growth on tf with gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# prevent information messages from tensorflow (such as "cuda loaded" etc)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description="Trainer for Convolutional Mesh Autoencoders")
parser.add_argument('--name', default='bareteeth', help='facial_motion| lfw ')
parser.add_argument('--data', default='data/bareteeth', help='facial_motion| lfw ')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train (default: 2)')
parser.add_argument('--eval_frequency', type=int, default=200, help='eval frequency')
parser.add_argument('--filter', default='chebyshev5', help='filter')
parser.add_argument('--nz', type=int, default=8, help='Size of latent variable')
parser.add_argument('--lr', type=float, default=8e-3, help='Learning Rate')
parser.add_argument('--workers', type=int, default=4, help='number of data loading threads')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--viz', type=int, default=0, help='visualize while test')
parser.add_argument('--loss', default='l1', help='l1 or l2')
parser.add_argument('--mesh1', default='m1', help='for mesh interpolation')
parser.add_argument('--mesh2', default='m1', help='for mesh interpolation')

args = parser.parse_args()

# set random seed
np.random.seed(args.seed)
# dimension of latent variable
working_dir = ""  # TODO

num_features = [16, 16, 16, 32]  # number of conv filters per conv layer
polynom_orders = [6, 6, 6, 6]  # polynomial orders
num_latent = 8
batch_size = 64
num_epochs = 100
initial_epoch= 50
validation_frequency = 1

# load reference mesh file
date_print("Loading template mesh.")
template_mesh_path = "/home/oole/git/ma/coma/impl/data/template.obj"
template_mesh = Mesh(filename=template_mesh_path)

# downsampling factors at each stage of sampling
downsampling_factors = [4, 4, 4, 4]

# ----- Precompute stuff according to template mesh
# A Adjecency matrices,
# D Downsampling matrices
# U Upsampling matrices
## D/U Sampling mesh 4 times, mesh is sampled by factor of 4

date_print("Precomputing adjecency/downsampling/upsampling matrices and graph laplaciancs according to adjecency "
           "matrices based on templat mesh")

meshes, adjecency_matrices, downsampling_matrices, upsampling_matrices = mesh_sampling.generate_transformation_matrices(
    template_mesh, downsampling_factors)

adjecency_matrices = [x.astype('float32') for x in adjecency_matrices]  # convertType(adjecency_matrices)
downsampling_matrices = [x.astype('float32') for x in downsampling_matrices]
upsampling_matrices = [x.astype('float32') for x in upsampling_matrices]
p = [x.shape[0] for x in adjecency_matrices]

# A: 5023x5023, 1256x1256, 314x314, 79x79, 20x20
# D: 1256x5023, 314x1256, 79x314, 20x79
# U: 5023x1256, 1256x314, 314x79, 79x20
# p: 5023, 1256, 314, 79, 20

# L Computed graph laplacians, computed for adjencency matrices a in A
L_1 = graph.laplacian(list(adjecency_matrices)[0])
laplacians = [graph.laplacian(matrix) for matrix in adjecency_matrices]
# L same dimensions as adjecency_matrices
# ----- Read dataset

mesh_data = meshdata.MeshData(number_val=100, train_file="/media/oole/Storage/Msc/processed-data/sliced" + '/train.npy',
                              test_file="/media/oole/Storage/Msc/processed-data/sliced" + '/test.npy',
                              reference_mesh_file=template_mesh_path)

x_train = mesh_data.vertices_train.astype('float32')
x_val = mesh_data.vertices_val.astype('float32')
x_test = mesh_data.vertices_test.astype('float32')

num_train = x_train.shape[0]
date_print("Training shape:   \t" + str(x_train.shape))
date_print("Validation shape: \t" + str(x_val.shape))
date_print("Test shape:       \t" + str(x_test.shape))

x_train = x_train[:-(x_train.shape[0] % batch_size)]
x_val = x_val[:-(x_val.shape[0] % batch_size)]
x_test = x_test[:-(x_test.shape[0] % batch_size)]
date_print("Training shape:   \t" + str(x_train.shape))
date_print("Validation shape: \t" + str(x_val.shape))
date_print("Test shape:       \t" + str(x_test.shape))
# ----- Parse Parameters
parameters = dict()
# Training configuration
num_input_features = int(x_train.shape[-1])

# Optimization TODO
# parameters['which_loss'] = args.loss
# parameters['nv'] = 784  # TODO check this value -> read from tempalte mesh
# parameters['regularization'] = 5e-4
# parameters['dropout'] = 1  # TODO ? no dropout?
# parameters['learning_rate'] = args.lr
# parameters['decay_rate'] = 0.99
# parameters['momentum'] = 0.9
# parameters['decay_steps'] = num_train / parameters['batch_size']

# Model configuration
# model = models.coma(L=L, D=D, U=U, **parameters)
coma_model = coma_ae(num_input_features=num_input_features,
                     num_features=num_features,
                     laplacians=laplacians[:-1],
                     downsampling_transformations=downsampling_matrices,
                     upsampling_transformations=upsampling_matrices,
                     Ks=polynom_orders,
                     num_latent=num_latent, batch_size=batch_size)
coma_model.compile(loss=keras.losses.MeanAbsoluteError(),
                   optimizer=keras.optimizers.Adam(3e-4), metrics=[keras.metrics.MeanAbsoluteError()])

if len(os.listdir("/home/oole/coma-model/checkpoint")) > 1:
    coma_model.load_weights("/home/oole/coma-model/checkpoint/")

save_callback = keras.callbacks.ModelCheckpoint(filepath="/home/oole/coma-model/checkpoint/",
                                                save_weights_only=True,
                                                monitor='accuracy',
                                                save_best_only=False,
                                                verbose=1)


tensorboard_callback = keras.callbacks.TensorBoard(log_dir="/home/oole/coma-model/tensorboard/")

coma_model.fit(x_train, x_train,
               batch_size=batch_size,
               epochs=num_epochs,
               shuffle=True,
               validation_freq=validation_frequency,
               validation_data=(x_val, x_val), callbacks=[save_callback, tensorboard_callback], initial_epoch=initial_epoch)

result = coma_model.predict(x_test, batch_size=batch_size)
print(result.shape)
mesh_util.visualizeSideBySide(original=x_test, prediction=result, number_of_meshes=10, mesh_data=mesh_data)

print("Result visualization")
# ----- Create model
