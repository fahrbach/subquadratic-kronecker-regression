# subquadratic-kronecker-regression

## Files
- `kronecker_regression_main.py` -- Implementation of `FastKroneckerRegression` and `DJSSW19`.
- `run_all_kronecker_regression.py` -- Wrapper to generate Figure 1 and Table 1.
- `tucker_als.py` -- Library with `KronMatMul` and `TuckerALS` implementations.
- `tucker_decomposition_experiments.py` -- Main file for running Tucker decomposition with command line arguments.
- `run_all_tensor_decomposition.py` -- Wrapper to run all Tucker decomposition experiments.
- `tensor_data_handler.py` -- Handler class to read different tensor data into `np.ndarray` format.
- `data/` -- Contains instructions on how to generate all of our datasets.
- `output/` -- Contains the output logs for all of our Kronecker regression and tensor decomposition experiments.
- `plots/` -- Contains code to generate Figure 1.

## Instructions
- `$ python3 run_all_kronecker_regression.py`
- `$ python3 run_all_tensor_decomposition.py`
