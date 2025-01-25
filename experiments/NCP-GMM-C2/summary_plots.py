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

exp_path = pathlib.Path("experiments/NCP-GMM-C2/C2PDE")
# exp_path = pathlib.Path("experiments/NCP-GMM-C2/C6MI")
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
    # "PMD/spectral_norm/test",
    # "PMD/mse/test",
    'PMD/invariance_err/test',
    'PMD/mse/test',
    # '||k(x,y) - k_r(x,y)||/test',
    ]
# sort metrics to plot
metrics_to_plot = sorted(metrics_to_plot)
log_scale_metrics = [
    # "PMD/mse/test"
    "PMD/mse/test",
    ]

df_melted = df.melt(id_vars=["model", "train_samples", "gmm.n_kernels"],
                    value_vars=metrics_to_plot,
                    var_name="metric",
                    value_name="value")
df_sorted = df_melted.sort_values(by="model")
# Define the metrics to plot

# Create a FacetGrid with rows for gmm.n_kernels and columns for each metric
sns.despine()

pallete = dict(ENCP="darkcyan", NCP="midnightblue", DRF="firebrick", IDRF="lightsalmon")

g = sns.FacetGrid(df_sorted,
                  row="gmm.n_kernels",
                  col="metric",
                  hue="model",
                  palette=pallete,
                  height=3,
                  aspect=1.0,
                  sharey=False,
                  sharex=True)

# Map the lineplot to the FacetGrid
g.map(sns.lineplot, "train_samples", "value",
      markers=True,
      errorbar=lambda x: (x.min(), x.max()),
      )

# Add legends, axis labels, and titles
g.add_legend()
g.set_axis_labels("No. training samples", "")
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
train_ratio = 0.7
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2,  # len(num_train_ratio_vals),
                       figsize=(7, 7),
                       sharey=True,
                       sharex=True,
                       gridspec_kw={'wspace': 0, 'hspace': 0},
                       dpi=150
                       )
df_sub = df[(df["train_samples_ratio"] == train_ratio)]

for model, ax in zip(["NCP", "ENCP", "DRF", "IDRF"], axes.flat):
    df_sub_model = df_sub[df_sub["model"] == model]
    data = [np.load(p / "npmi_data.npz") for p in df_sub_model["run_path"]]
    if len(data) == 0:
        continue
    pmi_gt = np.concatenate([d["pmd_gt"] for d in data])
    pmi = np.concatenate([d["pmd_pred"] for d in data])
    # sns.scatterplot(x=pmi_gt, y=pmi, ax=ax, alpha=0.5, color=pallete[model])
    sns.regplot(x=pmi_gt, y=pmi, ax=ax, marker="x", color=pallete[model], ci=99, scatter_kws={"alpha": 0.5})
    # Plot the ideal regession line
    max_y = pmi_gt.max()
    sns.lineplot(x=[0, max_y], y=[0, max_y], ax=ax, alpha=0.5, color="black")


# Add transparent gridlines to all axes
for ax in axes.flat:
    ax.grid(True, linestyle='-', alpha=0.1)
    # ax.set_yscale("symlog", linthresh=1e-3)
f.show()
#
#
# # Combine data for all models into a single DataFrame
# combined_data = []
#
# # for model in ["NCP", "ENCP", "DRF", "IDRF"]:
# # for model in ["NCP", "ENCP"]:
# for model in ["DRF", "ENCP"]:
#     df_sub_model = df_sub[df_sub["model"] == model]
#     data = [np.load(p / "npmi_data.npz") for p in df_sub_model["run_path"]]
#     if len(data) == 0:
#         continue
#     pmi_gt = np.concatenate([d["npmi_gt"] for d in data])
#     pmi = np.concatenate([d["npmi"] for d in data])
#     npmi_err = pmi_gt - pmi
#     model_data = pd.DataFrame({
#         "npmi_gt":  pmi_gt,
#         "npmi_err": pmi,
#         "model":    model
#         })
#     combined_data.append(model_data)
#
# if len(combined_data) == 0:
#     print("No data to plot")
#     exit(0)
# combined_df = pd.concat(combined_data)
#
# # Create the joint plot
# sns.set_theme(style="ticks")
# g = sns.jointplot(
#     data=combined_df,
#     x="npmi_gt", y="npmi_err", hue="model",
#     kind="kde",
#     levels=10,
#     clip=[[-1, 1], [-1, 1]],
#     # kind="hist",
#     # Set linewidth of the contour lines
#     linewidth=0.5,
#     fill=False,
#     cmap=sns.cubehelix_palette(as_cmap=True),
#     # bw_adjust=0.6,
#     )
# # Add gridlines
# g.ax_joint.grid(True, linestyle='-', alpha=0.3)
#
# plt.show()
