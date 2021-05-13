from psbody.mesh import MeshViewers, Mesh

def visualizeSideBySide(original, prediction, number_of_meshes, mesh_data):
    viewer = MeshViewers(shape=(2,number_of_meshes), titlebar="Above:Original, Below: Prediction")
    for i in range(number_of_meshes):
        viewer[1][i].set_dynamic_meshes([mesh_data.vec2mesh(original[i])])
        viewer[0][i].set_dynamic_meshes([mesh_data.vec2mesh(prediction[i])])

