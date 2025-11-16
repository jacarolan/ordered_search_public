Run `python3 lp.py --help` for help on how to use the solver or try running `python3 lp.py 3 56 1000`. 

To illustrate the three main parameters of the solver consider the above example `python3 lp.py 3 56 1000`. This will run the solver looking for a k = 3 query exact algorithm for searching a list of n = 56 elements. The grid against which nonnegativity of the polynomials will be checked consists initially of 1000 points. The algorithmic is iterative and after each iteration it may extend the grid with additional points. By default it exports the grids in json files to exports/grids/. 

This run will eventually terminate when the solver finds polynomials corresponding to a 3-query algorithm for searching 56 element lists. If a feasible solution has been found the solver will output something like

```
Solver finished. Time elapsed: 0.0
 The problem status is:  optimal
 Betas = [0.00681477 0.00681477]
 Syndrome = [ True  True  True  True]
 Minimal eval: 1.000000e+08
 Number of local minima found below 0: 0
```

When run on an infeasible instance (eg. `python3 lp.py 3 57 1000`) the solver will terminate with something like: 

```
Solver finished. Time elapsed: 0.07
 The problem status is:  optimal
 Betas = [-0.00082493 -0.00082493]
```

Here the negative beta values indicate that even the best possible polynomials are negative at some grid points. This witnesses that there's no 3-query algorithm for searching a 57-element list exactly. 


### Results

- The file `witnesses/grids_5_7266_5500_10.json` contains a grid for which the LP has negative optimal objective value, certifying that there is no 5-query translation-invariant algorithm for searching a list of 7266 elements. To verify this grid run `python3 lp.py 5 7266 5500 --grid witnesses/grids_5_7266_5500_10.json` (note this may take a while).
- The coefficients of the polynomials corresponding to a 5-query algorithm for searching a list of 7265 are stored in the file `witnesses/poly_coeffs_5_7265_5500_17.csv`.
