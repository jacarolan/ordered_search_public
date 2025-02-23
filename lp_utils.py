import timeit
import os
if os.name == 'posix':
    import resource
    from sys import platform
import numpy as np
import cvxpy as cp 
from scipy.linalg import block_diag

import verification_utils as vu 
import export_utils as eu 

SOLVE_EXIT_CODE_INCONCLUSIVE = 0
SOLVE_EXIT_CODE_ALL_POS = 1
SOLVE_EXIT_CODE_NEGATIVE_BETA = 2



class LPHelper:
    def __init__(self, k, N, positivity_grid_size, skip_initial_prune, dual_val_cutoff, use_odd_trick, solver, verbose, cull_offending_locs):
        self._k = k
        self._N = N
    
        self._positivity_grid_size = positivity_grid_size
        self._positivity_grids = None

        self._skip_initial_prune = skip_initial_prune
        self._dual_val_cutoff = dual_val_cutoff
        self._use_odd_trick = use_odd_trick
        self._solver = solver
        self._verbose = verbose
        self._cull_offending_locs = cull_offending_locs

        self._I = np.eye(N)
        self._V = block_diag([1], np.fliplr(np.eye(N-1)))

        self._plots_export_dir = None
        self._coefficients_export_dir = None

        if not self._use_odd_trick:
            self.__set_chebys_at_constraints()
        
        self.__set_coeffs_of_first_and_last_polys()
        self.__reset_coeff_values()
        self.__allocate_coefficients()
        self.__allocate_betas()



    def set_coefficients_export_dir(self, path):
        self._coefficients_export_dir = path
    


    def set_plots_export_dir(self, path):
        self._plots_export_dir = path



    def set_grids_export_dir(self, path):
        self._grids_export_dir = path



    def set_betas_export_dir(self, path):
        self._betas_export_dir = path



    def __set_coeffs_of_first_and_last_polys(self):
        self._coeffs_first_poly = np.zeros(self._N)
        self._coeffs_first_poly[0] = 1
        for i in range(1, self._N):
            self._coeffs_first_poly[i] = 2*(self._N - i)/self._N

        self._coeffs_last_poly = np.zeros(self._N)
        self._coeffs_last_poly[0] = 1 



    def __allocate_betas(self):
        self._betas = cp.Variable((self._k - 1,1))

    

    def get_value_of_betas(self):
        return self._betas.value



    def __allocate_coefficients(self):
        if self._use_odd_trick:
            nr_polys_to_allocate_variables_for = (self._k-1) // 2 
        else: 
            nr_polys_to_allocate_variables_for = self._k-1
        print("Allocating variables for the coefficients of " + str(nr_polys_to_allocate_variables_for) + " polynomials.")

        self._coeffs = cp.Variable((nr_polys_to_allocate_variables_for, self._N))



    def __set_chebys_at_constraints(self):
        constraints_grid = np.cos(np.linspace(0, np.pi, self._N+1))
        self._chebys_at_constraints = self.get_chebys_on_grid(constraints_grid, self._N)



    def get_chebys_on_grid(self, grid):
        chebys_on_grid = np.zeros((len(grid), self._N))
        chebys_on_grid[:,0] = 1
        chebys_on_grid[:,1] = grid
        for j in range(2, self._N): 
                chebys_on_grid[:, j] = 2 * (grid * chebys_on_grid[:, j-1]) - chebys_on_grid[:, j-2]

        return chebys_on_grid



    def __generate_positivity_constraints(self):
        chebys_on_pos_grids = list(map(lambda grid: self.get_chebys_on_grid(grid), self._positivity_grids))
        constraints = []

        for t in range(1, self._k):
            constraints += [chebys_on_pos_grids[t-1] @ self.get_coeffs(t) >= self._betas[t-1]]
        return constraints



    def get_coeffs(self, t, by_value = False):
        if t == 0:
            return self._coeffs_first_poly 
        elif t == self._k: 
            return self._coeffs_last_poly
        
        if self._use_odd_trick and (t % 2 == 1):
            coeffs_prev = self.get_coeffs(t-1, by_value)
            coeffs_next = self.get_coeffs(t+1, by_value)
            res = 1/2 * (coeffs_prev + coeffs_next) - 1/2 * self._V @ (coeffs_prev - coeffs_next)
            return res
        
        if self._use_odd_trick:
            idx = (int) (t / 2 - 1) # t != 0, q and t is even at this stage
        else: 
            idx = t-1 

        if by_value:
            return self._coeffs[idx, :].value 
        else:
            return self._coeffs[idx, :]
        


    def __generate_equality_constraints_no_trick(self):
        constraints = [] 
        for t in range(0, self._k):
            evals_curr = self._chebys_at_constraints @ self.get_coeffs(t) 
            evals_next = self._chebys_at_constraints @ self.get_coeffs(t+1)

            if t % 2 == 0:
                constraints += [evals_curr[1:self._N+1:2] == evals_next[1:self._N+1:2]]
            else:
                constraints += [evals_curr[0:self._N+1:2] == evals_next[0:self._N+1:2]]
        return constraints 



    def __generate_equality_constraints_odd_trick(self):
        constraints = [] 
        for t in range(0, self._k):
            if t % 2 == 0 and t > 0:
                coeffs_curr = self.get_coeffs(t)
                constraints += [coeffs_curr[0] == 1]

        if self._k % 2 == 0:
            return constraints
        
        coeffs_curr = self.get_coeffs(self._k-1) 
        coeffs_next = self.get_coeffs(self._k)
        constraints += [(self._I + (-1)**(t+1) * self._V) @ (coeffs_next - coeffs_curr) == 0]

        return constraints 
    


    def __set_equality_constraints(self):
        self._equality_constraints = self.__generate_equality_constraints()

    

    def __generate_equality_constraints(self):
        if self._use_odd_trick:
            return self.__generate_equality_constraints_odd_trick()
        else:
            return self.__generate_equality_constraints_no_trick()
        


    def __set_positivity_constraints(self):
        if self._positivity_grids == None: 
            self.__init_uniform_positivity_grids()
        
        self._positivity_constraints = self.__generate_positivity_constraints()



    def __init_uniform_positivity_grids(self):
        self._positivity_grids = [np.cos(np.linspace(0, np.pi, self._positivity_grid_size))] * (self._k-1)
        


    def prune_positivity_grids(self): 
        self._positivity_grids = self.get_relevant_locs(self._dual_val_cutoff)



    def extend_positivity_grid_with_offending_locs(self):
        (_, new_grid_locs, _) = self.get_offending_locs()

        cull_logic = lambda arr: arr if(len(arr) < 20) else arr[0::5]

        if(self._cull_offending_locs):
            new_grid_locs = list(map(cull_logic, new_grid_locs))

        self._positivity_grids = vu.merge_grids(self._positivity_grids, new_grid_locs)



    def get_problem_status(self):
        return self._problem.status
    


    def print_size_metrics(self): 
        size_metrics = self._problem.size_metrics
        print(" Size params as follows (scalar vars, eq constr, ineq constr):", size_metrics.num_scalar_variables, size_metrics.num_scalar_eq_constr, size_metrics.num_scalar_leq_constr)



    def solve_problem(self):
        t_elapsed = timeit.Timer(lambda: self.__solve_problem()).timeit(number = 1)
        self.__print_solution_diagnostics(t_elapsed)
        if self.__betas_have_negative_entry():
            return SOLVE_EXIT_CODE_NEGATIVE_BETA

        (all_positive, positivity_syndrome, min_eval, offending_locs) = self.verify_solution()
        self.__print_verification_info(min_eval, offending_locs, positivity_syndrome)
        if all_positive:
            return SOLVE_EXIT_CODE_ALL_POS
        
        return SOLVE_EXIT_CODE_INCONCLUSIVE



    def verify_solution(self):
        (min_eval, offending_locs, _) = self.get_offending_locs()

        all_positive = True
        positivity_syndrome = [True] + [False] * (self._k - 1) + [True]
        for t in range(0, self._k - 1):
            if len(offending_locs[t]) == 0: 
                positivity_syndrome[t+1] = True
            else: 
                all_positive = False 
        
        return (all_positive, positivity_syndrome, min_eval, offending_locs)



    def __solve_problem(self):
        self.__reset_coeff_values()
        self.__init_problem()
        self.__print_problem_diagnostics()
        self._problem.solve(solver=self._solver, verbose=self._verbose)        



    def __print_problem_diagnostics(self):
        grid_sizes = list(map(len, self._positivity_grids))
        print("Starting solver...")
        print(" Using positivity grids of sizes " + str(grid_sizes))
        self.print_size_metrics()



    def __print_solution_diagnostics(self, t_elapsed):
        if os.name == 'posix':
            multiplier = 1e9 if platform == "darwin" else 1e6
            max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(" Maximal memory usage " + str(round(max_memory / multiplier, 4)) + " GB")
        print("Solver finished. Time elapsed: " + str(round(t_elapsed, 2)))
        print(" The problem status is: ", self.get_problem_status())
        print(" Betas = " + str(np.transpose(self.get_value_of_betas())[0]))



    def __betas_have_negative_entry(self):
        betas = self.get_value_of_betas()
        return bool(len(betas[betas < 0]))



    def __print_verification_info(self, min_eval, offending_locs, positivity_syndrome):        
        print(" Syndrome = " + str(np.transpose(positivity_syndrome)))
        print(" Minimal eval: " + "{:e}".format(min_eval))
        print(" Number of local minima found below 0: " + str(sum(list(map(len, offending_locs)))))



    def __init_problem(self):
        self.__set_positivity_constraints()
        self.__set_equality_constraints()

        self._problem = cp.Problem(cp.Maximize(cp.min(self._betas) + 1e-6 * cp.sum(self._betas)), 
                        [*self._equality_constraints, *self._positivity_constraints]
                        )
        


    def get_coeff_values(self):
        if self._coeff_values.size == 0: 
            self.__set_coeff_values()

        return self._coeff_values 
    


    def __set_coeff_values(self):
        self._coeff_values = np.zeros((self._k+1, self._N))
        for t in range(self._k+1):
            self._coeff_values[t, :] = self.get_coeffs(t, True)



    def __reset_coeff_values(self):
        self._coeff_values = np.array([])
    


    def get_relevant_locs(self, relative_error):
        return vu.get_relevant_locs(self._positivity_grids, self._positivity_constraints, relative_error)
    


    def get_offending_locs(self):
        coeff_values = self.get_coeff_values()
        return vu.get_offending_locs_by_derivative(coeff_values[1:-1], 0)



    def __iterate(self, iter_nr, prune = False):
        print("\n\n *** ITERATION #" + str(iter_nr) + " *** ")

        return_code = self.solve_problem()
        self.export_betas(iter_nr)
        
        if return_code == SOLVE_EXIT_CODE_ALL_POS:
            self.export_plots(iter_nr)
            self.export_coefficients(iter_nr)
            return 0
        
        if return_code == SOLVE_EXIT_CODE_NEGATIVE_BETA:
            return 0 
        
        if prune: 
            self.prune_positivity_grids()
        else:
            self.extend_positivity_grid_with_offending_locs()
            
        self.export_positivity_grids(iter_nr)
        return 1 
        
    

    def resume_search_from(self, grids_file_path, prune_first, reset_counter, max_iter):
        start_iter = self.import_positivity_grids(grids_file_path)
        if reset_counter: 
            start_iter = 0
        print("Resuming search with grids from " + grids_file_path + " starting with iteration " + str(start_iter+1))
        
        if prune_first: 
            print("Starting with a pruning step.")
            res = self.__iterate(start_iter + 1, True)
            if res == 0:
                return 0 

        for iter_nr in range(start_iter + 1 + int(prune_first), max_iter + 1):
            res = self.__iterate(iter_nr)
            if res == 0:
                return 0 



    def find_certified_solution(self, max_iter = 40): 
        do_prune = not self._skip_initial_prune
        res = self.__iterate(1, prune = do_prune)
        if res == 0:
            return 0

        for iter_nr in range(2, max_iter+1): 
            res = self.__iterate(iter_nr)
            if res == 0:
                return 0 

        return 1



    def export_coefficients(self, iter_nr):
        params_suffix = self.__get_params_suffix(iter_nr)
        eu.export_array(self._coefficients_export_dir, "poly_coeffs" + params_suffix, self.get_coeff_values())



    def export_plots(self, iter_nr):
        params_suffix = self.__get_params_suffix(iter_nr)
        eu.export_poly_coeffs_plot(self._plots_export_dir, self.get_coeff_values(), params_suffix)
        eu.export_polynomials_plot(self._plots_export_dir, self.get_coeff_values(), params_suffix)



    def export_positivity_grids(self, iter_nr):
        params_suffix = self.__get_params_suffix(iter_nr)
        eu.export_grids(self._grids_export_dir,  params_suffix, self._positivity_grids)



    def import_positivity_grids(self, filepath):
        self._positivity_grids = eu.import_grids(filepath)
        return eu.extract_iter_nr_from_filepath(filepath)



    def export_betas(self, iter_nr):
        params_suffix = self.__get_params_suffix(iter_nr)
        eu.export_array(self._betas_export_dir, "betas" + params_suffix, self._betas.value)



    def __get_params_suffix(self, iter_nr):
        return eu.get_assemble_params_suffix(self._k, self._N, self._positivity_grid_size, not self._use_odd_trick, iter_nr)