# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 31/03/25
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plot_quantiles(y_c, low_q, up_q, ax=None, label="pred", alpha=0.05, gt=True, quantile_color="red", color=None):
    """Plots the empirical quantiles and the predicted quantiles.

    Args:
        y_c (torch.Tensor): The data to compute the ground truth quantiles.
        low_q (np.ndarray): The lower quantiles predicted by the model.
        up_q (np.ndarray): The upper quantiles predicted by the model.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis are created.
        label (str, optional): The label for the predicted quantiles. Defaults to "pred".
    """
    # Compute the empirical quantiles GT
    q_up_gt, q_lo_gt = np.quantile(y_c, 1 - (alpha / 2), axis=0), np.quantile(y_c, alpha / 2, axis=0)

    y_c = y_c
    y0_low, y1_low = low_q[0], low_q[1]
    y0_up, y1_up = up_q[0], up_q[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(y_c[:, 0], y_c[:, 1], alpha=0.05, color=color)
    # ax.scatter(y_c[:100, 0], y_c[:100, 1], color="green")
    ax.scatter(low_q[0], low_q[1], color=quantile_color)
    ax.scatter(up_q[0], up_q[1], color=quantile_color)

    # Add rectangles for the predicted quantiles
    pred_rect = Rectangle(
        (y0_low, y1_low), y0_up - y0_low, y1_up - y1_low, edgecolor=quantile_color, facecolor="none", alpha=0.8
    )
    ax.add_patch(pred_rect)
    ax.text(y0_low, y1_up * 1.05, label, color=quantile_color, verticalalignment="bottom", fontweight="bold")

    # Plot the GT
    ax.scatter(q_lo_gt[0], q_lo_gt[1], color="black")
    ax.scatter(q_up_gt[0], q_up_gt[1], color="black")

    # Add rectangles for the ground truth quantiles
    if gt:
        gt_rect = Rectangle(
            (q_lo_gt[0], q_lo_gt[1]),
            q_up_gt[0] - q_lo_gt[0],
            q_up_gt[1] - q_lo_gt[1],
            edgecolor="black",
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(gt_rect)
        ax.text(
            q_lo_gt[0],
            q_lo_gt[1] * 0.8,
            f"GT{int((1-alpha)*100):d}%",
            color="black",
            verticalalignment="top",
            horizontalalignment="left",
            fontweight="bold",
        )
    return fig, ax
