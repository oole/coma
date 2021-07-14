from psbody.mesh import MeshViewers
import numpy as np
import readchar
from util.log_util import date_print, hint_print
import model.model
from util import mesh_util

def play_with_latent_space(model: model.model.coma_ae, mesh_data, batch_size, mesh=None):
    if mesh is not None:
        # use specified mesh
        print("TODO")
    else:
        #use template mesh
        x_reference = np.full((batch_size, 5023, 3), mesh_data.reference_mesh.v)

    latent_representation = model.encode(x_reference).numpy()
    viewer = MeshViewers(shape=(1, 1), titlebar="Interactive Latent Representation")
    date_print(str(latent_representation[0]))
    while(1):
        # date_print("Change latent representation +(1,2,3,4,5,6,7,8,9) -(q,w,e,r,t,y,u,i).")

        input_key = readchar.readchar()

        if input_key == "1":
            latent_representation[0][0] = 1.01*latent_representation[0][0]
        elif input_key == "2":
            latent_representation[0][1] = 1.01*latent_representation[0][1]
        elif input_key == "3":
            latent_representation[0][2] = 1.01*latent_representation[0][2]
        elif input_key == "4":
            latent_representation[0][3] = 1.01*latent_representation[0][3]
        elif input_key == "5":
            latent_representation[0][4] = 1.01*latent_representation[0][4]
        elif input_key == "6":
            latent_representation[0][5] = 1.01*latent_representation[0][5]
        elif input_key == "7":
            latent_representation[0][6] = 1.01*latent_representation[0][6]
        elif input_key == "8":
            latent_representation[0][7] = 1.01*latent_representation[0][7]

        elif input_key == "q":
            latent_representation[0][0] = 0.99*latent_representation[0][0]
        elif input_key == "w":
            latent_representation[0][1] = 0.99*latent_representation[0][1]
        elif input_key == "e":
            latent_representation[0][2] = 0.99*latent_representation[0][2]
        elif input_key == "r":
            latent_representation[0][3] = 0.99*latent_representation[0][3]
        elif input_key == "t":
            latent_representation[0][4] = 0.99*latent_representation[0][4]
        elif input_key == "y":
            latent_representation[0][5] = 0.99*latent_representation[0][5]
        elif input_key == "u":
            latent_representation[0][6] = 0.99*latent_representation[0][6]
        elif input_key == "i":
            latent_representation[0][7] = 0.99*latent_representation[0][7]
        elif input_key == "\x1b":
            # escape
            break
        else:
            hint_print("(1,2,3,4,5,6,7,8) to increase values (*1.01), (q,w,e,r,t,y,u,i) to decrease values(*0.99)")
        # decode
        #visualize
        date_print(str(latent_representation[0]))
        decoded = model.decode(latent_representation).numpy()
        viewer[0][0].set_dynamic_meshes([mesh_data.vec2mesh(decoded[0])])


def sample_latent_space(model: model.model.coma_ae, mesh_data, batch_size, mesh=None):
    if mesh is not None:
        esh = np.full((batch_size, 5023, 3), mesh.v)
        # use specified mesh
        print("TODO")
    else:
        #use template mesh
        mesh = np.full((batch_size, 5023, 3), mesh_data.reference_mesh.v)

    latent_representation = model.encode(mesh).numpy()[0]
    samples = []
    for i in range(len(latent_representation)):
        component_samples = []
        for j in range(-4,5,1):
            new_rep =  (1 + 0.3 * j) * latent_representation[0]
            component_samples.append(new_rep)
        component_samples = np.array(component_samples)
    samples = np.array(samples)
    decoded = []
    for i in range(len(samples)):
        decoded.append(model.decode(samples[i]).numpy())