# Revisiting Optimism and Model Complexity in the Wake of Overparameterized Machine Learning

## Files

The scripts are organized in the following way.
The parameters of the simulations are described at the beginning of each of the file.
If there are multiple parameter settings for different experiments, the details are included after comment '# Setting different parameters for different experiments'.

### Experiment 1: theory
- `run_simu_mn2ls.py`: Ridgeless with varying numbers of features included (Figure 1).


- `run_simu_ridge.py`: Ridge with varying ridge regularization levels (Figure 11)
- `run_simu_ridgeless.py`: Ridgeless with varying data aspect ratios (Figure 12)

- `run_simu_lasso_iso.py`: Lasso with varying lasso regularization levels on isotropic features (Figure 13)
- `run_simu_lassoless_iso.py`: Lassoless with varying lasso regularization levels on isotropic features (Figure 14)

### Experiment 2: experiments

- `run_simu_lasso.py`: Lasso with varying lasso regularization levels (Figures 4-5)
- `run_simu_rf.py`: Random forest with varying numbers of trees and leaves (Figure 6)
- `run_simu_knn.py`: kNN with varying numbers of neighbors (Figures 15-16)
- `run_simu_random_features.py`: Ridgeless on random features with varying data aspect ratios (Figure 17)


### Experiment 3: Degrees of freedom comparisons
This requires running the following scripts used in the previous experiments with different parameters (Figures 7-10):
- `run_simu_ridge.py`
- `run_simu_rf.py`
- `run_simu_knn.py`

### Visualization
- `Plot.ipynb`: visualize the results of the simulations.

### Utilities
- `compute_risk.py`: compute the theoretical and empirical risk of various estimators.
- `generate_data.py`: generate data for the simulations.
- `plot.py`: utility function for plotting for `Plot.ipynb`.



## Computation details
All the experiments are run on Ubuntu 22.04.4 LTS using 12 cores and 128 GB of RAM.

The estimated time to run all experiments is roughly 12 hours.


## Dependencies

Package | Version
--- | ---
h5py | 3.1.0
joblib | 1.4.0
matplotlib | 3.4.3
numpy | 1.20.3
pandas | 1.3.3
python | 3.8.12
scikit-learn | 1.3.2
sklearn_ensemble_cv | 0.2.3
scipy | 1.10.1
statsmodels | 0.13.5
tqdm | 4.62.3