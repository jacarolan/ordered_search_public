Run `python3 lp.py --help` for help on how to use the solver or try running `python3 lp.py 3 56 1000`.


### Results

- The file `witnesses/grids_5_7266_5500_10.json` contains a grid for which the LP has negative optimal objective value, certifying that there is no 5-query translation-invariant algorithm for searching a list of 7266 elements.
- The coefficients of the polynomials corresponding to a 5-query algorithm for searching a list of 7265 are stored in the file `witnesses/poly_coeffs_5_7265_5500_17.csv`. 