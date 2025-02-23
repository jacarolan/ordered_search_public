import argparse
import cvxpy as cp
from lp_utils import LPHelper



EXPORTS_DIR = "exports/"
COEFFS_EXPORT_SUBDIR = "coeffs/" 
PLOTS_EXPORT_SUBDIR = "plots/" 
GRIDS_EXPORT_SUBDIR = "grids/"
BETAS_EXPORT_SUBDIR = "betas/"


parser = argparse.ArgumentParser()
parser.add_argument("query_count", help="The number of queries.", type=int)
parser.add_argument("instance_size", help="The size of the OSP instance.", type=int)
parser.add_argument("grid_size", help="The size of the grid for the non-negativity constraints.", type=int)
parser.add_argument("--dual-cutoff", help="The size of the grid for the non-negativity constraints.", type=float)
parser.add_argument("--max-iter", help="The maximal number of iterations to be used when searching for a certified solution.", type=int)
parser.add_argument("--solver", help="Choose which solver to use.", type=str)
parser.add_argument("--verbose", help="Add flag to run CVXPY in verbose mode.", action='store_true')
parser.add_argument("--no-tricks", help="Add flag to avoid using even/odd trick.", action='store_true')
parser.add_argument("--generate-plots", help="Add flag to generate and export plots.", action='store_true')
parser.add_argument("--grids", help="Specify relative path to file containing non-negativity grids to resume search from.", type=str)
parser.add_argument("--prune", help="To be used in conjunction with --grid. The search will resume with an initial pruning step.", action='store_true')
parser.add_argument("--reset-counter", help="To be used in conjunction with --grid. The search will start the iteration counter at 1.", action='store_true')
parser.add_argument("--skip-initial-prune", help="Add flag to skip the pruning step after the first iteration.", action="store_true")
parser.add_argument("--cull-offending-locs", help="Add flag so that grids are only extended with every 3rd offending location. This may help with numerical stability.", action='store_true')
args = parser.parse_args()

solver = None
if args.solver == "CVXOPT":
    solver = cp.CVXOPT
if args.solver == "SCS":
    solver = cp.SCS
if args.solver == "CLARABEL":
    solver = cp.CLARABEL
if args.solver == "COPT":
    solver = cp.COPT    
if solver == None: 
    args.solver = "MOSEK"
    solver = cp.MOSEK


dual_val_cutoff = 1e-10
if args.dual_cutoff != None:
    dual_val_cutoff = args.dual_cutoff 

max_iter = 80
if args.max_iter != None:
    max_iter = args.max_iter


helper = LPHelper(args.query_count, args.instance_size, args.grid_size, args.skip_initial_prune, dual_val_cutoff, not args.no_tricks, solver, args.verbose, args.cull_offending_locs)
helper.set_coefficients_export_dir(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR)
helper.set_plots_export_dir(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR)
helper.set_grids_export_dir(EXPORTS_DIR + GRIDS_EXPORT_SUBDIR)
helper.set_betas_export_dir(EXPORTS_DIR + BETAS_EXPORT_SUBDIR)

if args.grids != None: 
    helper.resume_search_from(args.grids, args.prune, args.reset_counter, max_iter)
else: 
    helper.find_certified_solution(max_iter)