import argparse
import numpy as np
from data import meshdata
from util.log_util import date_print

parser = argparse.ArgumentParser()
parser.add_argument("--prediction", default="results/lr8e3_bareteeth_bs16_210713_1256_result.npy",
                    help="The path to the predictions.npy")
parser.add_argument("--data-dir", default="/media/oole/Storage/Msc/processed-data/bareteeth",
                    help="The path to the data")
parser.add_argument("--error-dir", default="error", help="The directory where the computed errors should be stored")
parser.add_argument("--template-mesh", default="data/template.obj", help="The path to the template mesh")

args = parser.parse_args()
predictions = np.load(args.prediction)
date_print("Calculating errors for " + args.prediction + " - Dir: " + args.data_dir)

mesh_data = meshdata.MeshData(number_val=100, train_file=args.data_dir + '/train.npy',
                              test_file=args.data_dir + '/test.npy',
                              reference_mesh_file=args.template_mesh, fit_pca=True)

# CoMA
predicted_vertices = (predictions * mesh_data.std) + mesh_data.mean

original_vertices = (mesh_data.vertices_test[:predictions.shape[0]] * mesh_data.std) + mesh_data.mean

# we want millimeters
predicted_vertices_mm = predicted_vertices * 1000
original_vertices_mm = original_vertices * 1000

model_error = np.sqrt(np.sum((predicted_vertices_mm - original_vertices_mm) ** 2, axis=2))
model_error_mean = np.mean(model_error)
model_error_std = np.std(model_error)
model_error_median = np.median(model_error)

date_print("CoMA Error - Mean: " + str(model_error_mean) + ", Std: " + str(model_error_std) + ", Median: " + str(
    model_error_median))
# PCA
pca_prediction = mesh_data.pca.inverse_transform(mesh_data.pca.transform(
    np.reshape(mesh_data.vertices_test, (mesh_data.vertices_test.shape[0], mesh_data.n_vertex * 3))))

pca_vertices = (np.reshape(pca_prediction, (pca_prediction.shape[0], mesh_data.n_vertex, 3)) * mesh_data.std) + mesh_data.mean
pca_vertices_mm = pca_vertices * 1000

original_vertices = (mesh_data.vertices_test * mesh_data.std) + mesh_data.mean
original_vertices_mm = original_vertices * 1000

pca_error = np.sqrt(np.sum((pca_vertices_mm - original_vertices_mm) ** 2, axis=2))
pca_error_mean = np.mean(pca_error)
pca_error_std = np.std(pca_error)
pca_error_median = np.median(pca_error)

date_print("PCA Error - Mean: " + str(pca_error_mean) + ", Std: " + str(pca_error_std) + ", Median: " + str(
    pca_error_median))


