from matplotlib import pyplot as plt
from os import makedirs
import numpy as np
import json 

CSV_DELIMITER = ', '
DISPLAY_GRID_SIZE = 3000



def export_poly_coeffs_plot(export_dir, coeffs, params_suffix):
    makedirs(export_dir, exist_ok=True)
    (q_plus_one, N) = coeffs.shape
    for i in range(0, q_plus_one):
        plt.plot(coeffs[i, :], label="q_" + str(i))
    plt.legend()
    fig_name = "SDP_polynomial_coeffs" + params_suffix + ".png"
    plt.savefig(export_dir + fig_name)
    plt.clf()



def export_polynomials_plot(export_dir, coeffs, params_suffix):
    makedirs(export_dir, exist_ok=True)
    (q_plus_one, N) = coeffs.shape
    xs = np.linspace(0, 2*np.pi, DISPLAY_GRID_SIZE) 
    display_grid = np.cos(xs)
    chebys_on_display_grid = np.zeros((DISPLAY_GRID_SIZE, N))
    chebys_on_display_grid[:,0] = 1
    chebys_on_display_grid[:,1] = display_grid
    for i in range(2, N): 
        chebys_on_display_grid[:,i] = 2 * (display_grid * chebys_on_display_grid[:, i-1]) - chebys_on_display_grid[:, i-2]

    for i in range(0, q_plus_one):
        plt.plot(xs, chebys_on_display_grid @ coeffs[i, :], label="q_" + str(i))
    plt.legend()
    fig_name = "SDP_polynomials" + params_suffix + ".png"
    plt.savefig(export_dir + fig_name)
    plt.clf()



def export_array(export_dir, filename_without_extension, arr):
    makedirs(export_dir, exist_ok=True)
    np.savetxt(export_dir + filename_without_extension + ".csv", np.transpose(arr), fmt="%+1.6f", delimiter = CSV_DELIMITER)

    

def import_poly_coeffs(dir, params_suffix):
    return np.genfromtxt(dir + "poly_coeffs" + params_suffix + ".csv", delimiter = CSV_DELIMITER)



def export_grids(export_dir, params_suffix, grids):
    makedirs(export_dir, exist_ok = True)
    with open(export_dir + "grids" + params_suffix + ".json", 'w') as f:
        json.dump(list(map(lambda arr: arr.tolist(), grids)), f)



def import_grids(rel_path):
    with open(rel_path) as f:
        raw_grids = json.load(f)

    grids = []
    for grid in raw_grids:
        grids += [np.array(grid)]
    return grids



def export_coordinate_plot(export_dir, filename, coords):
    makedirs(export_dir, exist_ok=True)
    plt.scatter(coords, np.zeros_like(coords))
    plt.xlim(-1, 1)
    plt.ylim(-0.1, 0.1)
    plt.savefig(export_dir + filename)
    plt.clf()



def show_plots_for_grids(grids):
    for grid in grids: 
        plt.scatter(grid, np.zeros_like(grid))
        plt.xlim(-1, 1)
        plt.ylim(-0.1, 0.1)
        plt.show()
        plt.clf()



def get_assemble_params_suffix(k, N, grid_size, no_tricks, iteration_nr):
    suffix = "_" + str(k) + "_" + str(N) + "_" + str(grid_size)
    if no_tricks:
        suffix += "_no-tricks"

    suffix += "_" + str(iteration_nr)
    return suffix



def extract_iter_nr_from_filepath(path):
    path_without_ext = path.split(".")[0]
    return int(path_without_ext.split("_")[-1])