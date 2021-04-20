import os
import copy
import time
import numpy as np
import argparse
import mesh_sampling
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers

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

#set random seed
np.random.seed(args.seed)
# dimension of latent variable
nz = args.nz

# load reference mesh file
mesh = Mesh(filename="/home/oole/git/ma/coma/impl/data/template.obj")


# downsampling factors at each stage of sampling
ds_factors = [4, 4, 4, 4]

# A Adjecency matrices,
# D Downsampling matrices
# U Upsampling matrices
## D/U Sampling mesh 4 times, mesh is sampled by factor of 4

M, A, D, U = mesh_sampling.generate_transformation_matrices(mesh, ds_factors)

print("Mesh vertices:")
i = 1
for mes in M:
    print("Mesh " + str(i))
    print("Vertices: " + str(len(mes.v)))
    print("Faces: " + str(len(mes.f)))
    i += 1

viewer = MeshViewers(shape=[1,len(M)], titlebar="Meshes")
i = 0
for mes in M:
    viewer[0][i].set_static_meshes([mes])
    i += 1
