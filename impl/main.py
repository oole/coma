import os
import json
import numpy as np
import argparse

from util import mesh_sampling, mesh_util, latent_magic
from util.log_util import date_print
from psbody.mesh import MeshViewers, Mesh
from data import meshdata
import util.graph_util as graph
from model.model import coma_ae
import model.model_util as model_util
from tensorflow import keras
import tensorflow as tf
import model.tboard as tboard

## experimental config for memory growth on tf with gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# prevent information messages from tensorflow (such as "cuda loaded" etc)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description="Convolutional Mesh Autoencoder written for Tensorflow 2")
parser.add_argument("--name", default="default-run-name",
                    help="The name of the run (used for checkpoints and tensorboard")
parser.add_argument("--data-folder", default="data/sliced",
                    help="Path to the data folder containing train.npy and test.npy")
parser.add_argument("--automatic-run-name", type=bool, default=False,
                    help="Whether the run name should be automatically constructed (default is False)")
parser.add_argument("--batch-size", type=int, default=16, help="The batch size to be used (default is 16)")
parser.add_argument("--num-epochs", type=int, default=300, help="The number of training epochs (default is 300")
parser.add_argument("--initial-epoch", type=int, default=0,
                    help="The initial epoch, useful for continue training on an existing run (default 0)")
parser.add_argument("--latent-vector-length", type=int, default=8, help="The size of the latent vector (default is 8)")
parser.add_argument("--validation-frequency", type=int, default=10, help="The validation frequency")
parser.add_argument("--learning-rate", type=float, default=8e-3, help="The learning rate (default is 8e-3")
parser.add_argument("--random-seed", type=int, default=2, help="The random seed (default is 8)")
parser.add_argument("--template-mesh", default="data/template.obj", help="Path to the template mesh")
parser.add_argument("--mode", default="train", help="The mode to run in (train, test, latent")
parser.add_argument("--sanity-check", type=bool, default=False, help="Whether or not sanity check should be performed")
parser.add_argument("--coma-model-dir", default="/home/oole/coma-model",
                    help="The directory holding checkpoints and tensorboard (Such as /home/oole/coma-model/tensorboard or /home/oole/coma-model/checkpoint)")
parser.add_argument("--visualize-during-training", type=bool, default=False,
                    help="Whether the meshes should be visualized in tensorboard during training")

args = parser.parse_args()

# Set parsed arguments
run_name = args.name
np.random.seed(args.random_seed)
num_latent = args.latent_vector_length
batch_size = args.batch_size
num_epochs = args.num_epochs
initial_epoch = args.initial_epoch
validation_frequency = args.validation_frequency

base_coma_model_dir = args.coma_model_dir
base_data_folder = args.data_folder

template_mesh_path = args.template_mesh

learning_rate = args.learning_rate  # 1e-2  # done TODO; original was 8e-3

num_features = [16, 16, 16, 32]  # number of conv filters per conv layer
polynom_orders = [6, 6, 6, 6]  # polynomial orders

# used for local model:
load_checkpoint = base_coma_model_dir + "/checkpoint/" + run_name

save_checkpoint = load_checkpoint
# used for local model:
tensorboard_dir = base_coma_model_dir + "/tensorboard/" + run_name + "/"

# load reference mesh file
date_print("Loading template mesh.")

# template_mesh_path = "/workspace/coma/impl/data/template.obj"
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

if not os.path.exists("computed/meshes-save.npy"):
    meshes, adjecency_matrices, downsampling_matrices, upsampling_matrices = mesh_sampling.generate_transformation_matrices(
        template_mesh, downsampling_factors)
    if not os.path.exists("computed"):
        os.makedirs("computed")
    np.save("computed/meshes-save.npy", meshes)
    np.save("computed/adjecency_matrcies-save.npy", adjecency_matrices)
    np.save("computed/downsampling_matrices-save.npy", downsampling_matrices)
    np.save("computed/upsampling_matrices-save.npy", upsampling_matrices)
else:
    meshes = np.load("computed/meshes-save.npy", allow_pickle=True)
    adjecency_matrices = np.load("computed/adjecency_matrcies-save.npy", allow_pickle=True)
    downsampling_matrices = np.load("computed/downsampling_matrices-save.npy", allow_pickle=True)
    upsampling_matrices = np.load("computed/upsampling_matrices-save.npy", allow_pickle=True)

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

mesh_data = meshdata.MeshData(number_val=100, train_file=base_data_folder + '/train.npy',
                              test_file=base_data_folder + '/test.npy',
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

# Training parameters and regularization:
decay_rate = 0.99  # done
momentum = 0.9  # done
decay_steps = num_train  # / batch_size  # done
regularization = 5e-4

# Model configuration
# model = models.coma(L=L, D=D, U=U, **parameters)
coma_model = coma_ae(num_input_features=num_input_features,
                     num_features=num_features,
                     laplacians=laplacians[:-1],
                     downsampling_transformations=downsampling_matrices,
                     upsampling_transformations=upsampling_matrices,
                     Ks=polynom_orders,
                     num_latent=num_latent, batch_size=batch_size,
                     regularization=regularization)

coma_model.compile(loss=keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                   optimizer=keras.optimizers.SGD(
                       learning_rate=model_util.get_learning_rate_decay_schedule(learning_rate, decay_rate,
                                                                                 decay_steps),
                       momentum=momentum), metrics=[keras.metrics.MeanAbsoluteError()])

if os.path.exists(load_checkpoint) and len(os.listdir(load_checkpoint)) > 1:
    coma_model.load_weights(load_checkpoint + "/coma_model")

# mesh_util.pageThroughMeshes(mesh_data.vertices_train.astype('float32'), mesh_data)

if args.mode == "train":
    # store parameters
    parameter_dir = base_coma_model_dir + "/model-parameters"
    if not os.path.exists(parameter_dir):
        os.makedirs(parameter_dir)
    parameter_file = parameter_dir + "/" + args.name + "_parameters.json"
    with open(parameter_file, 'w') as file:
        save_params = dict()
        save_params['batch_size'] = args.batch_size
        save_params['name'] =args.name
        save_params['num_epochs'] = args.num_epochs
        save_params['validation-frequency'] = args.validation_frequency
        save_params['num_filters'] = num_features
        save_params["polynom-orders"] = polynom_orders
        save_params['random-seed'] = args.random_seed
        save_params['learning-rate'] = args.learning_rate
        save_params['decay-steps'] = decay_steps
        save_params['decay-rate'] = decay_rate
        save_params['momentum'] = momentum
        save_params['regularization'] = regularization
        save_params['num-latent'] = num_latent
        save_params['data-folder'] = base_data_folder
        date_print(str(save_params))
        json.dump(save_params, file)
    save_callback = keras.callbacks.ModelCheckpoint(filepath=save_checkpoint + "/coma_model",
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    save_best_only=False,
                                                    verbose=1)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    if args.visualize_during_training:
        tensorboard_mesh_indices = [0, 82, 94, 109, 159, 227, 342, 373, 454, 553, 591, 617, 747, 880, 980, 1008, 1079,
                                    1223,
                                    1524, 1642, 1780, 1807, 1973, 2029, 2155, 2202, 2381, 2459, 2544, 2631, 2766, 2902,
                                    2975, 3060, 3153, 3285, 3354, 3535, 3771, 3848, 4033, 4196, 4339, 4514, 4664, 4790,
                                    4858, 5339, 5384, 5513, 5562, 5699, 5756, 5847, 6045, 6313, 6499, 6723, 6781, 7133,
                                    7201, 7353, 7509, 7671]
        tensorboard_mesh_indices = tensorboard_mesh_indices[:batch_size]
        tensorboard_meshes = np.array([x_train[i] for i in tensorboard_mesh_indices])
        mesh_callback = tboard.MeshCallback(tb_meshes=tensorboard_meshes, template_mesh=template_mesh,
                                            batch_size=batch_size,
                                            log_dir=tensorboard_dir, mesh_data=mesh_data)

        coma_model.fit(x_train, x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       validation_freq=validation_frequency,
                       validation_data=(x_val, x_val), callbacks=[save_callback, tensorboard_callback, mesh_callback],
                       initial_epoch=initial_epoch)
    else:
        coma_model.fit(x_train, x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       validation_freq=validation_frequency,
                       validation_data=(x_val, x_val), callbacks=[save_callback, tensorboard_callback],
                       initial_epoch=initial_epoch)
elif args.mode == "test":
    result = coma_model.evaluate(x=x_test, y=x_test, batch_size=batch_size)
    metric_names = coma_model.metrics_names
    print(metric_names[0] + ": " + str(result[0]) + " -- " + metric_names[1] + ": " + str(result[1]))
    result = coma_model.predict(x_test, batch_size=batch_size)
    print(result.shape)
    mesh_util.visualizeSideBySide(original=x_test, prediction=result, number_of_meshes=10, mesh_data=mesh_data)

    tensorboard_mesh_indices = [0, 82, 94, 109, 159, 227, 342, 373, 454, 553, 591, 617, 747, 880, 980, 1008, 1079, 1223,
                                1524, 1642, 1780, 1807, 1973, 2029, 2155, 2202, 2381, 2459, 2544, 2631, 2766, 2902,
                                2975, 3060, 3153, 3285, 3354, 3535, 3771, 3848, 4033, 4196, 4339, 4514, 4664, 4790,
                                4858, 5339, 5384, 5513, 5562, 5699, 5756, 5847, 6045, 6313, 6499, 6723, 6781, 7133,
                                7201, 7353, 7509, 7671]
    tensorboard_mesh_indices = tensorboard_mesh_indices[:batch_size]
    tensorboard_meshes = np.array([x_train[i] for i in tensorboard_mesh_indices])

    tensorboard_meshes = tensorboard_meshes[:batch_size]
    tb_result = coma_model.predict(tensorboard_meshes, batch_size=batch_size)
    mesh_util.visualizeSideBySide(original=x_test, prediction=tb_result, number_of_meshes=10, mesh_data=mesh_data)
elif args.mode == "latent":
    latent_magic.play_with_latent_space(model=coma_model, mesh_data=mesh_data, batch_size=batch_size)
elif args.mode == "sample":
    # Todo sample from latent space
    print("todo.")

if args.sanity_check:
    x_reference = np.full((batch_size, 5023, 3), mesh_data.reference_mesh.v)
    x_result = coma_model.predict(x_reference, batch_size=batch_size)
    mesh_util.visualizeSideBySide(original=x_reference, prediction=x_result, number_of_meshes=1, mesh_data=mesh_data)
