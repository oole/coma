import os
import copy
import time
import numpy as np
import argparse
from util import mesh_sampling
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
from data import meshdata
import util.graph_util as graph
import model.coma as coma

## experimental config for memory growth on tf with gpu
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], true)
# prevent information messages from tensorflow (such as "cuda loaded" etc)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
nz = args.nz

# load reference mesh file
print("Loading template mesh.")
template_mesh_path = "/home/oole/git/ma/coma/impl/data/template.obj"
template_mesh = Mesh(filename=template_mesh_path)

# downsampling factors at each stage of sampling
downsampling_factors = [4, 4, 4, 4]

# ----- Precompute stuff according to template mesh
# A Adjecency matrices,
# D Downsampling matrices
# U Upsampling matrices
## D/U Sampling mesh 4 times, mesh is sampled by factor of 4

print("Precomputing adjecency/downsampling/upsampling matrices \n"
      "and graph laplaciancs according to adjecency matrices based on templat mesh")

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

# ----- Parse Parameters
parameters = dict()
# Training configuration
parameters['dir_name'] = args.name
parameters['num_epochs'] = args.num_epochs
parameters['batch_size'] = args.batch_size
parameters['eval_frequency'] = args.eval_frequency

parameters['filter'] = args.filter
parameters['brelu'] = 'b1relu'
parameters['pool'] = 'poolwT'
parameters['unpool'] = 'poolwT'

parameters['F_0'] = int(x_train.shape[2])  # input dimension
parameters['F'] = [16, 16, 16, 32]  # number of conv filters per conv layer
parameters['K'] = [6, 6, 6, 6]  # polynomial orders
parameters['p'] = p  # pooling size, corresponds to size of adjecency matrix
parameters['nz'] = [nz]  # size of latent vector

# Optimization
parameters['which_loss'] = args.loss
parameters['nv'] = 784  # TODO check this value -> read from tempalte mesh
parameters['regularization'] = 5e-4
parameters['dropout'] = 1  # TODO ? no dropout?
parameters['learning_rate'] = args.lr
parameters['decay_rate'] = 0.99
parameters['momentum'] = 0.9
parameters['decay_steps'] = num_train / parameters['batch_size']

# Model configuration
# model = models.coma(L=L, D=D, U=U, **parameters)
model = coma.ComaModel(laplacians=laplacians, downsampling_matrices=downsampling_matrices,
                       upsampling_matrices=upsampling_matrices, **parameters)

# ----- Create model
