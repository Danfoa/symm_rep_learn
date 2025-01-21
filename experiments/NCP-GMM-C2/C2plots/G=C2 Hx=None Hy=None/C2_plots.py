# Created by danfoa at 21/01/25
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from lightning import seed_everything
from omegaconf import OmegaConf

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.examples.symm_GMM import get_symmetry_group

exp_path = pathlib.Path("experiments/NCP-GMM-C2/C2plots")
print(exp_path)
assert exp_path.exists()
# Find all "test_metrics.csv" paths in the experiment directory
test_metrics_paths = list(exp_path.rglob("**/test_metrics.csv"))
config_paths = [p.parent / ".hydra/config.yaml" for p in test_metrics_paths]
overrides_paths = [p.parent / ".hydra/overrides.yaml" for p in test_metrics_paths]
# Load all experiment configurations
run_cfgs = [OmegaConf.load(str(p)) for p in config_paths]

# Define the experiments hyperparameters to consider
exp_hparams = ["model", "gmm.n_samples", "gmm.n_kernels"]

# Create pandas dataframe with the exp_hparams in the columns and a column of test_metrics.csv paths as the index
df = pd.DataFrame(columns=exp_hparams)
for i, (cfg, metrics_path) in enumerate(zip(run_cfgs, test_metrics_paths)):
    for hparam in exp_hparams:
        val = OmegaConf.select(cfg, hparam)
        # Index dataframe by run number
        df.loc[i, hparam] = val
    metrics_vals = pd.read_csv(metrics_path)
    # Put all metrics in the experiment run. Each metric is a column
    for col in metrics_vals.columns:
        df.loc[i, col] = metrics_vals[col].values[0] # Test metrics are scalar values

metric_names = list(df.columns)
print(metric_names)
# Seaborn plots for ["PMD/spectral_norm/test", "NPMI/mse/test", "PMD/mse/test"] vs "gmm.n_samples" with hue="model"
# Create a new DataFrame with "metric" and "value" columns
df_melted = df.melt(id_vars=["model", "gmm.n_samples", "gmm.n_kernels"],
                    value_vars=["PMD/spectral_norm/test", "NPMI/mse/test", "PMD/mse/test"],
                    var_name="metric",
                    value_name="value")
df_sorted = df_melted.sort_values(by="gmm.n_kernels")

# Define the metrics to plot
metrics_to_plot = ["PMD/spectral_norm/test", "NPMI/mse/test", "PMD/mse/test"]

# Create a FacetGrid with rows for gmm.n_kernels and columns for each metric
g = sns.FacetGrid(df_sorted,
                  row="gmm.n_kernels",
                  col="metric",
                  hue="model",
                  height=3,
                  aspect=1.0,
                  sharey=False, sharex=True)

# Map the lineplot to the FacetGrid
g.map(sns.lineplot, "gmm.n_samples", "value")

# Add legends, axis labels, and titles
g.add_legend()
g.set_axis_labels("Number of GMM Samples", "")
g.set_titles(row_template="{row_name} kernels", col_template="{col_name}")

g.fig.show()
# Show the plot