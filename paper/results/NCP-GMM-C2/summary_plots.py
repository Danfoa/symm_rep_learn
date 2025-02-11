# Created by danfoa at 21/01/25
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

FIG_HEIGHT = 2.2
FONT_SIZE_LEGEND = 7
FONT_SIZE_TICKS = 9
FONT_SIZE_AX_LABELS = 10
FONT_SIZE_TITLES = 7
ASPECT_RATIO = 1.25
# Initialize an empty DataFrame to hold all experiments
df = pd.DataFrame()

exp_labels = ["C2", "C6", "D6", "Ico"]
for exp_label in exp_labels:
    exp_path = pathlib.Path(f"experiments/NCP-GMM-C2/{exp_label}")

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
    df_exp = pd.DataFrame(columns=exp_hparams)
    for i, (cfg, metrics_path) in enumerate(zip(run_cfgs, test_metrics_paths)):
        for hparam in exp_hparams:
            val = OmegaConf.select(cfg, hparam)
            # Index dataframe by run number
            df_exp.loc[i, hparam] = val
        metrics_vals = pd.read_csv(metrics_path)
        # Put all metrics in the experiment run. Each metric is a column
        for col in metrics_vals.columns:
            df_exp.loc[i, col] = metrics_vals[col].values[0]  # Test metrics are scalar values
        df_exp.loc[i, "run_path"] = metrics_path.parent

    # Add the exp_label to the DataFrame
    df_exp["exp_label"] = exp_label
    # Append the current experiment DataFrame to the all_experiments_df
    df = pd.concat([df, df_exp], ignore_index=True)

df = df.sort_values(by="model")
df["train_samples"] = df["gmm.n_total_samples"] * df["train_samples_ratio"]
# Sort by model:
df = df.sort_values(by="model")
metric_names = list(df.columns)
print(metric_names)

# Create a new DataFrame with "metric" and "value" columns
metrics_to_plot = [
    # "PMD/spectral_norm/test",
    # "PMD/mse/test",
    "PMD/invariance_err/test",
    "PMD/mse/test",
    # '||k(x,y) - k_r(x,y)||/test',
]
# sort metrics to plot
metrics_to_plot = sorted(metrics_to_plot)
log_scale_metrics = [
    # "PMD/mse/test"
    "PMD/mse/test",
]

df_melted = df.melt(
    id_vars=["model", "train_samples", "exp_label"], value_vars=metrics_to_plot, var_name="metric", value_name="value"
)
df_sorted = df_melted.sort_values(by="model")
# Define the metrics to plot

# Create a FacetGrid with rows for gmm.n_kernels and columns for each metric
sns.despine()

pallete = dict(ENCP="darkcyan", NCP="midnightblue", DRF="firebrick", IDRF="lightsalmon")

g = sns.FacetGrid(
    df_sorted,
    row="metric",
    col="exp_label",
    col_order=exp_labels,  # Ensure column order respects exp_labels
    hue="model",
    palette=pallete,
    height=FIG_HEIGHT,
    margin_titles=False,
    aspect=ASPECT_RATIO,
    sharey=False,
    sharex=False,
)
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
g.add_legend(
    title="Model",
    fontsize=FONT_SIZE_LEGEND,
    title_fontsize=FONT_SIZE_LEGEND,
    label_order=["ENCP", "NCP", "IDRF", "DRF"],
)
g.set_axis_labels("No. training samples", "", fontsize=FONT_SIZE_AX_LABELS)
g.set_titles(row_template="{row_name} kernels", col_template="{col_name}", fontsize=FONT_SIZE_TITLES)  # Titles
# Set log scale for specific metrics
for ax in g.axes.flat:
    title = ax.get_title()
    for log_metric in log_scale_metrics:
        if log_metric in title:
            ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, linestyle="-", alpha=0.2, axis="y")

g.tight_layout()
g.set_xlabels(fontsize=FONT_SIZE_AX_LABELS)
g.figure.subplots_adjust(wspace=0.1, hspace=0.5)
g.fig.savefig(fname=pathlib.Path("experiments/NCP-GMM-C2/") / "test_metrics.png", dpi=300)
plt.show(dpi=250)

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
