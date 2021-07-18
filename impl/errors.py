import argparse
import numpy as np
from data import meshdata
from util.log_util import date_print
import json
import os
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--prediction", default="results/lr8e3_sliced_coma_result.npy",
                    help="The path to the predictions.npy")
parser.add_argument("--data-dir", default="data/sliced",
                    help="The path to the data")
parser.add_argument("--error-dir", default="error", help="The directory where the computed errors should be stored")
parser.add_argument("--template-mesh", default="data/template.obj", help="The path to the template mesh")


def cumulative_error(error):
    num_bins = 1000
    num_vertices, x_error = np.histogram(error, bins=num_bins)
    x_error = np.convolve(x_error, 0.5 * np.ones(2), mode='valid')

    factor_error = 100 / error.shape[0]

    cumulative_num_vertices = np.cumsum(num_vertices)

    x_error_vector = np.zeros((num_bins + 1,))
    x_error_vector[1:] = x_error

    y_vec_error = np.zeros((num_bins + 1,))
    y_vec_error[1:] = factor_error * cumulative_num_vertices

    return x_error_vector, y_vec_error


def plot_error_over_vertices(coma_error, pca_error, vertices, path):
    coma_error = np.reshape(coma_error, (-1,))
    pca_error = np.reshape(pca_error, (-1,))

    x_coma, y_coma = cumulative_error(coma_error)
    x_pca, y_pca = cumulative_error(pca_error)

    plt.plot(x_pca, y_pca, label="PCA")
    plt.plot(x_coma, y_coma, label="CoMA")
    plt.ylabel('Vertices (percent)')
    plt.xlabel('Euclidean Error (mm)')

    plt.legend(loc='lower right')
    plt.xlim(0, 6)
    plt.grid(True)  # ,color='grey', linestyle='-', linewidth=0.5)
    plt.savefig(path)
    pass


def calculate_error(predicted_vertices, original_vertices):
    error = np.sqrt(np.sum((predicted_vertices - original_vertices) ** 2, axis=2))
    error_mean = np.mean(error)
    error_std = np.std(error)
    error_median = np.median(error)
    return error, error_mean, error_std, error_median


args = parser.parse_args()
predictions = np.load(args.prediction)
date_print("Calculating errors for " + args.prediction + " - Dir: " + args.data_dir)

mesh_data = meshdata.MeshData(number_val=100, train_file=args.data_dir + '/train.npy',
                              test_file=args.data_dir + '/test.npy',
                              reference_mesh_file=args.template_mesh, fit_pca=True)

# CoMA
date_print("Predicting using CoMA")
predicted_vertices = (predictions * mesh_data.std) + mesh_data.mean

original_vertices = (mesh_data.vertices_test[:predictions.shape[0]] * mesh_data.std) + mesh_data.mean

# we want millimeters
predicted_vertices_mm = predicted_vertices * 1000
original_vertices_mm = original_vertices * 1000

model_error, model_error_mean, model_error_std, model_error_median = calculate_error(predicted_vertices_mm,
                                                                                     original_vertices_mm)

date_print("CoMA Error - Mean: " + str(model_error_mean) + ", Std: " + str(model_error_std) + ", Median: " + str(
    model_error_median))
# PCA
date_print("Predicting using PCA")
pca_prediction = mesh_data.pca.inverse_transform(mesh_data.pca.transform(
    np.reshape(mesh_data.vertices_test, (mesh_data.vertices_test.shape[0], mesh_data.n_vertex * 3))))

pca_vertices = (np.reshape(pca_prediction,
                           (pca_prediction.shape[0], mesh_data.n_vertex, 3)) * mesh_data.std) + mesh_data.mean
pca_vertices_mm = pca_vertices * 1000

original_vertices = (mesh_data.vertices_test * mesh_data.std) + mesh_data.mean
original_vertices_mm = original_vertices * 1000

pca_error, pca_error_mean, pca_error_std, pca_error_median = calculate_error(pca_vertices_mm, original_vertices_mm)

date_print("PCA Error - Mean: " + str(pca_error_mean) + ", Std: " + str(pca_error_std) + ", Median: " + str(
    pca_error_median))

date_print("Saving error plot")
error_plot_path = args.error_dir + "/error_plot"
if not os.path.exists(error_plot_path):
    os.makedirs(error_plot_path)

plot_error_over_vertices(model_error, pca_error, original_vertices,
                         error_plot_path + "/" + os.path.basename(args.data_dir))

date_print("Storing errors.")
if not os.path.exists(args.error_dir):
    os.makedirs(args.error_dir)
# CoMA
coma_error_file = args.error_dir + "/" + "coma_" + os.path.basename(args.data_dir)
with open(coma_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['coma_model_error'] = model_error
    save_params['coma_model_error_mean'] = model_error_mean
    save_params['coma_model_error_std'] = model_error_std
    save_params['coma_model_error_median'] = model_error_median
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(coma_error_file + "_error.npy", model_error)
# PCA
pca_error_file = args.error_dir + "/" + "pca_" + os.path.basename(args.data_dir)
with open(pca_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['pca_model_error'] = pca_error
    save_params['pca_model_error_mean'] = pca_error_mean
    save_params['pca_model_error_std'] = pca_error_std
    save_params['pca_model_error_median'] = pca_error_median
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(pca_error_file + "_error.npy", pca_error)
