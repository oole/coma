import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.mesh import summary_v2 as mesh_summary
import numpy as np


class MeshCallback(keras.callbacks.Callback):
    """
    Takes the example mesh and visualizes it after every epoch.
    """

    def __init__(self, tb_meshes, template_mesh, batch_size, mesh_data, log_dir):
        super(MeshCallback, self).__init__()
        self.tb_meshes = tb_meshes
        self.template_mesh = template_mesh
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir + "mesh")
        self.mesh_data = mesh_data

    def on_epoch_begin(self, epoch, logs=None):
        prediction = self.model.predict(self.tb_meshes, batch_size=self.batch_size)

        with self.writer.as_default():


            # original
            for i in range(self.batch_size):
                tb_mesh_one = self.mesh_data.vec2mesh(self.tb_meshes[i])
                prediction_one = self.mesh_data.vec2mesh(prediction[i])
                # points with color code
                points_rec = prediction_one.v
                points_original = tb_mesh_one.v
                # Idea: elementwise euclidean distance, normalized, and then colorized on reconstruction.
                error = np.abs(np.linalg.norm(points_rec - points_original, axis=1))
                error_norm = (error - np.min(error)) / np.max(error)
                # Use this to color:
                error_color = np.array([(error * 255, (1 - error) * 255, 0) for error in error_norm])

                mesh_summary.mesh(name="original_mesh_" + str(i) + "_vis", vertices=np.expand_dims(points_original, 0),
                                  faces=np.expand_dims(tb_mesh_one.f, 0),
                                  colors=np.expand_dims(error_color, 0),
                                  step=epoch)
                # reconstruction
                mesh_summary.mesh(name="reconstructed_mesh_" + str(i) + "_vis", vertices=np.expand_dims(points_rec, 0),
                                  faces=np.expand_dims(prediction_one.f, 0),
                                  colors=np.expand_dims(error_color, 0),
                                  step=epoch)
            # faces = np.expand_dims(np.vstack((prediction_one.f, tb_mesh_one.v)), 0)

            # visualize error....
