# Created by danfoa at 21/01/25
import pathlib

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from lightning import seed_everything
from omegaconf import OmegaConf

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.examples.symm_GMM import get_symmetry_group

exp_path = pathlib.Path("experiments/NCP-GMM-C2/C6plots_final_final")
print(exp_path)
assert exp_path.exists(), f"Experiment path {exp_path.absolute()} does not exist"
# Find all "test_metrics.csv" paths in the experiment directory
test_metrics_paths = list(exp_path.rglob("**/test_metrics.csv"))
config_paths = [p.parent / ".hydra/config.yaml" for p in test_metrics_paths]
overrides_paths = [p.parent / ".hydra/overrides.yaml" for p in test_metrics_paths]
# Load all experiment configurations
run_cfgs = [OmegaConf.load(str(p)) for p in config_paths]

# Define the experiments hyperparameters to consider
exp_hparams = ["model", "gmm.n_total_samples", "train_samples_ratio", "gmm.n_kernels", "seed"]

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
        df.loc[i, col] = metrics_vals[col].values[0]  # Test metrics are scalar values
    df.loc[i, "run_path"] = metrics_path.parent

# WARNING: Hacky because of dir err
# df = df[df["gmm.n_total_samples"] == 20000]
df['train_samples'] = df["gmm.n_total_samples"] * df["train_samples_ratio"]
# ==============================================================================

# Sort by model:
df = df.sort_values(by="model")

metric_names = list(df.columns)
print(metric_names)
# Seaborn plots for ["PMD/spectral_norm/test", "NPMI/mse/test", "PMD/mse/test"] vs "gmm.n_samples" with hue="model"
# Create a new DataFrame with "metric" and "value" columns
metrics_to_plot = [
    "PMD/spectral_norm/test",
    "PMD/mse/test",
    '||k(x,y) - k_r(x,y)||/test',
    # 'E_p(x,y) k_r(x,y)/test',
    # 'E_p(x)p(y) k_r(x,y)^2/test',
    'PMD/equiv_err/test'
    ]
# sort metrics to plot
metrics_to_plot = sorted(metrics_to_plot)
log_scale_metrics = [
    # "PMD/mse/test"
    ]

df_melted = df.melt(id_vars=["model", "train_samples", "gmm.n_kernels"],
                    value_vars=metrics_to_plot,
                    var_name="metric",
                    value_name="value")
df_sorted = df_melted.sort_values(by="gmm.n_kernels")

# Define the metrics to plot

# Create a FacetGrid with rows for gmm.n_kernels and columns for each metric
g = sns.FacetGrid(df_sorted,
                  row="gmm.n_kernels",
                  col="metric",
                  hue="model",
                  height=3,
                  aspect=1.0,
                  sharey=False, sharex=True)

# Map the lineplot to the FacetGrid
g.map(sns.lineplot, "train_samples", "value", errorbar=lambda x: (x.min(), x.max()))

# Add legends, axis labels, and titles
g.add_legend()
g.set_axis_labels("Number of training samples", "")
g.set_titles(row_template="{row_name} kernels", col_template="{col_name}")
# Set log scale for specific metrics
for ax in g.axes.flat:
    title = ax.get_title()
    for log_metric in log_scale_metrics:
        if log_metric in title:
            ax.set_yscale("log")
    ax.set_xscale("log")
g.fig.show()
g.fig.savefig(fname=exp_path / "test_metrics.png", dpi=150)
# Show the plot

# ==============================================================================
num_kernel_vals = df["gmm.n_kernels"].unique()
num_train_ratio_vals = df["train_samples_ratio"].unique()
n_kernels = 3
train_ratio = 0.7

# Set up the matplotlib figure
f, axes = plt.subplots(4, 2, #len(num_train_ratio_vals),
                       figsize=(7, 7),
                       sharey=True,
                       sharex=True,
                       gridspec_kw={'wspace': 0, 'hspace': 0},
                       dpi=150
                       )

df_sub = df[(df["gmm.n_kernels"] == n_kernels) & (df["train_samples_ratio"] == train_ratio) & (df["seed"] == 0)]

# for row, model in enumerate(["NCP", "ENCP", "DRF", "IDRF"]):
#     df_sub_model = df_sub[df_sub["model"] == model]
#     data = [np.load(p / "npmi_data.npz") for p in df_sub_model["run_path"]]
#     if len(data) == 0:
#         continue
#     npmi_gt = np.concatenate([d["npmi_gt"] for d in data])
#     npmi = np.concatenate([d["npmi"] for d in data])
#     npmi_err = npmi_gt - npmi
#     ax = sns.kdeplot(
#         x=npmi_gt,
#         y=npmi_err,
#         fill=True,
#         cmap=sns.cubehelix_palette(as_cmap=True),
#         levels=10,
#         bw_adjust=0.6,
#         ax=axes[row, 0],
#         clip=[[-1,1],[-1,1]],
#         )
#     # Get the range of the color bar used in the ax
#     cbar = ax.collections[0].colorbar
# # Add transparent gridlines to all axes
# for ax in axes.flat:
#     ax.grid(True, linestyle='-', alpha=0.3)
#     # ax.set_yscale("symlog", linthresh=1e-3)
# f.show()
#

# Combine data for all models into a single DataFrame
combined_data = []

# for model in ["NCP", "ENCP", "DRF", "IDRF"]:
# for model in ["NCP", "ENCP"]:
for model in ["DRF", "ENCP"]:
    df_sub_model = df_sub[df_sub["model"] == model]
    data = [np.load(p / "npmi_data.npz") for p in df_sub_model["run_path"]]
    if len(data) == 0:
        continue
    npmi_gt = np.concatenate([d["npmi_gt"] for d in data])
    npmi = np.concatenate([d["npmi"] for d in data])
    npmi_err = npmi_gt - npmi
    model_data = pd.DataFrame({
        "npmi_gt": npmi_gt,
        "npmi_err": npmi,
        "model": model
        })
    combined_data.append(model_data)

if len(combined_data) == 0:
    print("No data to plot")
    exit(0)
combined_df = pd.concat(combined_data)


# Create the joint plot
sns.set_theme(style="ticks")
g = sns.jointplot(
    data=combined_df,
    x="npmi_gt", y="npmi_err", hue="model",
    kind="kde",
    levels=10,
    clip=[[-1, 1], [-1, 1]],
    # kind="hist",
    # Set linewidth of the contour lines
    linewidth=0.5,
    fill=False,
    cmap=sns.cubehelix_palette(as_cmap=True),
    # bw_adjust=0.6,
    )
# Add gridlines
g.ax_joint.grid(True, linestyle='-', alpha=0.3)

plt.show()