import numpy as np
import numpy.polynomial.chebyshev as cb

LARGE_NUMBER = 1e8
IMAGINARY_PART_CUTOFF = 1e-8



def get_offending_locs_by_grid(coeffs, verification_grid_size, multiplier = 1e-3):
    (q_plus_one, N) = coeffs.shape

    grid = np.cos(np.linspace(0, np.pi, verification_grid_size))
    chebys_on_grid = np.zeros((verification_grid_size, N))
    chebys_on_grid[:,0] = 1
    chebys_on_grid[:,1] = grid
    for i in range(2, N): 
        chebys_on_grid[:,i] = 2 * (grid * chebys_on_grid[:, i-1]) - chebys_on_grid[:, i-2]

    min_evals = []
    for t in range(1, q_plus_one-1): 
        evals_on_grid = chebys_on_grid @ coeffs[t,:]
        min_evals += [np.min(evals_on_grid)]

    offending_locs = []
    for t in range(1, q_plus_one-1):
        evals_on_grid = chebys_on_grid @ coeffs[t,:]
        threshold = min_evals[t-1] * multiplier 

        current_offending_idx = np.where(evals_on_grid < threshold)[0]
        offending_locs += [grid[current_offending_idx]]

    return (np.min(min_evals), offending_locs)



def print_grid_sizes_for_different_cutoffs(positivity_constraints, cutoff):
    cutoff = cutoff * 1e3 

    for _ in range(7):
        idxx = get_relevant_constr_idx_for_cutoff(positivity_constraints, cutoff)

        lens = []
        for idx in idxx:
            lens += [len(idx)]
        print(cutoff, lens)

        cutoff = cutoff / 10 



def get_relevant_locs(positivity_grids, positivity_constraints, cutoff):
    print_grid_sizes_for_different_cutoffs(positivity_constraints, cutoff)

    idxx = get_relevant_constr_idx_for_cutoff(positivity_constraints, cutoff)
    
    locs = []
    for (idx, grid) in zip(idxx, positivity_grids):
        locs += [grid[idx]]        
    return locs
        


def get_relevant_constr_idx_for_cutoff(positivity_constraints, dual_val_cutoff):
    relevant_constraint_idx = []

    for constraint_vec in positivity_constraints:
        relevant_constraint_idx += [list(np.argwhere(constraint_vec.dual_value > dual_val_cutoff).flatten())]

    return relevant_constraint_idx



def get_combined_dual_val_of_grid_idx(constraints, idxx):
    res = 0
    for (constraint_vec, idx) in zip(constraints, idxx):
        res += np.sum(constraint_vec.dual_value[idx])
    return res



def merge_grids(lefts, rights):
    merged_grids = []
    
    for (l, r) in zip(lefts, rights):
        merged_grids += [np.array(list(l) + list(r))]
    return merged_grids



def prune(grid, round = 5):
    return list(np.unique(np.round(grid, round)))



def cheb_extrema(poly_coeffs):
    deriv = cb.chebder(poly_coeffs)
    deriv_roots = cb.chebroots(deriv)
    return deriv_roots



def cheb_points_below_cutoff(poly_coeffs, cutoff=1e-10):
    extrema = cheb_extrema(poly_coeffs)
    r_real = extrema.real[np.abs(extrema.imag) < IMAGINARY_PART_CUTOFF]
    r_less = r_real[r_real < 1.0]
    relevant_extrema = np.concatenate((np.array([-1]), r_less[r_less > -1.0], np.array([1])))
    vals_at_relevant_extrema = cb.chebval(relevant_extrema, poly_coeffs)
    violating_inds = (vals_at_relevant_extrema < -cutoff)
    return (relevant_extrema[violating_inds], vals_at_relevant_extrema[violating_inds])



# Returns a boolean for whether chebyshev polynomial p is positive in the interval [-1, 1]
def cheb_is_positive(poly_coeffs, cutoff = 1e-10): 
    (extrema_locs, _) = cheb_points_below_cutoff(poly_coeffs, cutoff)
    return (len(extrema_locs) == 0)



def get_offending_locs_by_derivative(coeffs, cutoff = 1e-10):
    (q_plus_one, N) = coeffs.shape
    offending_locs = []

    min_evals = []
    vals_at_offending_locs = []

    for t in range(q_plus_one):
        (extrema_locs, vals_at_extrema) = cheb_points_below_cutoff(coeffs[t, :], cutoff)
        offending_locs += [extrema_locs]
        vals_at_offending_locs += [vals_at_extrema]
        if len(vals_at_extrema) != 0:
            min_evals += [np.min(vals_at_extrema)]
        else:
            min_evals += [LARGE_NUMBER]

    return (np.min(min_evals), offending_locs, vals_at_offending_locs)