# Created by danfoa at 21/01/25
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from omegaconf import OmegaConf

FIG_HEIGHT = 2.5
FONT_SIZE_LEGEND = 7
FONT_SIZE_TICKS = 7
FONT_SIZE_AX_LABELS = 10
FONT_SIZE_TITLES = 7
ASPECT_RATIO = 1.7


exp_path = pathlib.Path("experiments/eNCP_regression/CoM_sample_eff")
# exp_path = pathlib.Path("experiments/eNCP_regression/CoM_sample_efficiency_solo")
# exp_path = pathlib.Path("experiments/eNCP_regression/CoM_sample_efficiency_atlas_v4_final")
# exp_path = pathlib.Path("experiments/eNCP_regression/CoM_sample_efficiency_anymal_c")

print(exp_path)
assert exp_path.exists(), f"Experiment path {exp_path.absolute()} does not exist"
# Find all "test_metrics.csv" paths in the experiment directory
test_metrics_paths = list(exp_path.rglob("**/test_metrics.csv"))
config_paths = [p.parent / ".hydra/config.yaml" for p in test_metrics_paths]
overrides_paths = [p.parent / ".hydra/overrides.yaml" for p in test_metrics_paths]
# Load all experiment configurations
run_cfgs = [OmegaConf.load(str(p)) for p in config_paths]

# Define the experiments hyperparameters to consider
exp_hparams = ["model", "optim.train_sample_ratio", "seed", "architecture.residual_encoder", "gamma"]

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

print(df["model"].unique())
print(df["optim.train_sample_ratio"].unique())
# WARNING: Hacky because of dir err
# df = df[df["gmm.n_total_samples"] == 20000]
df["train_samples"] = 100000 * df["optim.train_sample_ratio"]
# Filter for architecture.residual_encoder=False
df = df[df["architecture.residual_encoder"] is False]
df = df[df["optim.train_sample_ratio"] < 0.7]
# if model is
# df = df[((df["model"] == 'NCP') | (df["model"] == 'ENCP')) & (df["gamma"] == 25) | (~df["model"].isin(['NCP', 'ENCP']))]
# Sort by model:
df = df.sort_values(by="model")
metric_names = list(df.columns)
print(metric_names)
# ==============================================================================
train_samples = df["train_samples"].unique()
print(f"Train samples: {sorted(train_samples)}")
# Remove 50000 from train samples
# Seaborn plots for ["PMD/spectral_norm/test", "NPMI/mse/test", "PMD/mse/test"] vs "gmm.n_samples" with hue="model"
# Create a new DataFrame with "metric" and "value" columns
metrics_to_plot = [
    "y_mse/test",
    # 'hg_mse/test',
    # 'com_lin_momentum/test',
    # 'com_ang_momentum/test',
    # 'kinetic_energy/test',
]
# sort metrics to plot
metrics_to_plot = sorted(metrics_to_plot)
log_scale_metrics = [
    # "PMD/mse/test"
    "y_mse/test",
    "com_lin_momentum/test",
    "com_ang_momentum/test",
    # 'kinetic_energy/test',
]

df_melted = df.melt(
    id_vars=["model", "train_samples"], value_vars=metrics_to_plot, var_name="metric", value_name="value"
)
df_sorted = df_melted.sort_values(by="model")
# Define the metrics to plot

# Create a FacetGrid with rows for gmm.n_kernels and columns for each metric
sns.despine()

pallete = dict(ENCP="darkcyan", NCP="midnightblue", MLP="firebrick", EMLP="lightsalmon")

g = sns.FacetGrid(
    df_sorted,
    # row="gmm.n_kernels",
    col="metric",
    hue="model",
    palette=pallete,
    height=FIG_HEIGHT,
    aspect=ASPECT_RATIO,
    sharey=False,
    sharex=True,
)
g.tight_layout()
# Map the lineplot to the FacetGrid
g.map(
    sns.lineplot,
    "train_samples",
    "value",
    markers=True,
    errorbar=lambda x: (x.min(), x.max()),
)
g.set_xticklabels(fontsize=FONT_SIZE_TICKS)  # Font size for x-axis tick labels
g.set_yticklabels(fontsize=FONT_SIZE_TICKS)  # Font size for y-axis tick labels
g.tight_layout(rect=[0.1, 0, 1, 1])
g.add_legend(
    title="Model",
    fontsize=FONT_SIZE_LEGEND,
    title_fontsize=FONT_SIZE_LEGEND,
    label_order=["ENCP", "NCP", "EMLP", "MLP"],
)
g.set_axis_labels("No. training samples", "", fontsize=FONT_SIZE_AX_LABELS)
g.set_titles(row_template="{row_name} kernels", col_template="{col_name}", fontsize=FONT_SIZE_TITLES)  # Titles


# SI unit formatting function
def si_formatter(val, pos):
    # Format the values in k, M, G, etc.
    if val >= 1e3 and val < 1e6:
        return f"{val / 1e3:.0f}k"
    elif val >= 1e6 and val < 1e9:
        return f"{val / 1e6:.0f}M"
    elif val >= 1e9:
        return f"{val / 1e9:.0f}G"
    else:
        return f"{val:.0f}"


# Set log scale for specific metrics
for ax in g.axes.flat:
    title = ax.get_title()
    for log_metric in log_scale_metrics:
        if log_metric in title:
            ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\mathbf{y}_{\text{mse}}$", fontsize=FONT_SIZE_AX_LABELS)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICKS)
    ax.tick_params(axis="both", which="minor", labelsize=FONT_SIZE_TICKS)
    ax.grid(True, linestyle="-", alpha=0.2, axis="y", which="both")

    # Customize minor ticks to display simple decimal notation instead of scientific notation
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2, 0.4, 0.6, 0.8], numticks=10))

    # Define a function to format the minor ticks
    def minor_tick_formatter(val, pos):
        return f"{val:.1f}"  # Convert to simple decimal notation

    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(minor_tick_formatter))

    # Use LogLocator for MINOR ticks with fractional subs
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.5, 0.75, 4], numticks=10))

    # Format ticks with SI notation (e.g., 5k, 10k, 50k)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: f"{val / 1e3:.0f}k" if val >= 1e3 else f"{val:.0f}")
    )
    ax.xaxis.set_minor_formatter(
        ticker.FuncFormatter(lambda val, pos: f"{val / 1e3:.0f}k" if val >= 1e3 else f"{val:.0f}")
    )


# g.fig.show()
# plt.subplots_adjust(left=0.25, top=0.95, bottom=0.1)
g.fig.savefig(fname=exp_path / "test_metrics.png", dpi=250)
plt.show(dpi=200)

# Show the plot
# %% ==============================================================================
unique_train_ratios = df["train_samples_ratio"].unique()
# Construct the dataframe holding train ratio information
df_pmd_all = pd.DataFrame(columns=["train_samples_ratio", "model", "pmd_gt", "pmd_pred"])

for train_ratio in sorted(df["train_samples_ratio"].unique()):  # Ensure train ratios are ordered
    df_sub = df[df["train_samples_ratio"] == train_ratio]

    for model in ["NCP", "ENCP", "DRF", "IDRF"]:
        df_sub_model = df_sub[df_sub["model"] == model]
        data = [np.load(p / "npmi_data.npz") for p in df_sub_model["run_path"]]
        if len(data) == 0:
            continue
        pmd_gt = np.concatenate([d["pmd_gt"] for d in data])
        pmd_pred = np.concatenate([d["pmd_pred"] for d in data])
        df_pmd_all = pd.concat(
            [
                df_pmd_all,
                pd.DataFrame(
                    {"train_samples_ratio": train_ratio, "model": model, "pmd_gt": pmd_gt, "pmd_pred": pmd_pred}
                ),
            ]
        )

# Compute limits for the plots
upper_limit = np.percentile(df_pmd_all["pmd_gt"], 95)  # 95th percentile
y_max_lim = np.percentile(df_pmd_all["pmd_gt"] - df_pmd_all["pmd_pred"], 99)  # 99th percentile
y_min_lim = np.percentile(df_pmd_all["pmd_gt"] - df_pmd_all["pmd_pred"], 1)  # 1st percentile

# Create a custom column for the residuals
df_pmd_all["residual"] = df_pmd_all["pmd_gt"] - df_pmd_all["pmd_pred"]

# Initialize a FacetGrid for train ratios and models
g = sns.FacetGrid(
    df_pmd_all,
    row="model",
    col="train_samples_ratio",
    height=3,
    aspect=1,
    sharex=True,
    sharey=True,
    xlim=(0, upper_limit),
    ylim=(y_min_lim, y_max_lim),
)
# Map a kernel density plot to the grid
g.map(
    sns.kdeplot,
    "pmd_gt",
    "residual",
    fill=True,
    levels=10,
    cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    clip=[[0, upper_limit], [y_min_lim, y_max_lim]],
)
# Use g.map to pot a horixontal line from 0 to upper_limit
g.map(plt.axhline, y=0, color="black", linestyle="-", linewidth=1, alpha=0.4)
# Adjust layout and save the figure
g.tight_layout()
g.savefig(fname=exp_path / "error_dist_facetgrid.png", dpi=250)
plt.show()
