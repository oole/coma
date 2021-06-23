from psbody.mesh import MeshViewers, Mesh
import readchar
from util.log_util import date_print, hint_print


def visualizeSideBySide(original, prediction, number_of_meshes, mesh_data):
    viewer = MeshViewers(shape=(2, number_of_meshes), titlebar="Above:Original, Below: Prediction")
    for i in range(number_of_meshes):
        viewer[1][i].set_dynamic_meshes([mesh_data.vec2mesh(original[i])])
        viewer[0][i].set_dynamic_meshes([mesh_data.vec2mesh(prediction[i])])


def pageThroughMeshes(meshes, mesh_data):
    viewer = MeshViewers(shape=(1, 1), titlebar="Page view")
    index = 0
    num_meshes = meshes.shape[0]
    while (1):

        # date_print("Change latent representation +(1,2,3,4,5,6,7,8,9) -(q,w,e,r,t,y,u,i).")

        input_key = readchar.readchar()
        if input_key == "n":
            index += 1
        elif input_key == "p":
            index -= 1
        elif input_key == "\x1b":
            # escape
            break
        else:
            hint_print("n for next page, p for previous page.")


        if (index >= num_meshes):
            index = num_meshes - 1
            hint_print("End reached")
        if (index <= 0):
            index = 0
            hint_print("Start reached")

        date_print("Viewing index: " + str(index))

        # decode
        # visualize
        viewer[0][0].set_dynamic_meshes([mesh_data.vec2mesh(meshes[index])])
