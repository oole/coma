import os
import copy
import time
import numpy as np
import argparse
import mesh_sampling
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers

parser = argparse.ArgumentParser(description="Trainer for Convolutional Mesh Autoencoders")
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')

args = parser.parse_args()

#set random seed
np.random.seed(args.seed)
# dimension of latent variable

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
