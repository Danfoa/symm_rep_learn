from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
from typing import Optional

import numpy as np
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


def plot_gt_and_quantiles(
    gt,
    q_low,
    q_high,
    pred: Optional[torch.Tensor | np.ndarray] = None,
    title_prefix="Dim",
    subtitles=None,
    fig=None,
    ncols=None,
    area_color="rgba(0, 191, 255, 0.2)",  # light blue
    outlier_color="red",
    legend=True,
    time_0=0,
    row_offset=0,
    col_offset=0,
    gt_line_color="green",
    pred_line_color="blue",
    gt_line_style="solid",
    pred_line_style="solid",
):
    """Plots predictions vs ground truth per dimension with shared legend group using Plotly.

    Args:
        gt (torch.Tensor or np.ndarray): Ground truth (time, dim)
        pred (torch.Tensor or np.ndarray): Predictions (time, dim)
        q_low (torch.Tensor or np.ndarray): Lower quantile (time, dim)
        q_high (torch.Tensor or np.ndarray): Upper quantile (time, dim)
        title_prefix (str): Fallback prefix for subplot titles.
        subtitles (list of str): Optional list of titles per dimension.
        title (str): Title of the entire figure.
    """
    gt = gt.cpu().numpy() if hasattr(gt, "cpu") else gt
    q_low = q_low.cpu().numpy() if hasattr(q_low, "cpu") else q_low
    q_high = q_high.cpu().numpy() if hasattr(q_high, "cpu") else q_high
    pred_np = None
    if pred is not None:
        pred_np = pred.cpu().numpy() if hasattr(pred, "cpu") else pred

    time = np.arange(gt.shape[0])
    dim = gt.shape[1]
    n_cols = min(3, dim) if ncols is None else ncols
    n_rows = int(np.ceil(dim / n_cols))

    if fig is None:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.055,
            subplot_titles=[
                subtitles[i] if subtitles and i < len(subtitles) else f"{title_prefix} {i}" for i in range(dim)
            ],
        )

    for i in range(dim):
        row = (i) // n_cols + 1
        col = (i) % n_cols + 1

        # Ground Truth line
        fig.add_trace(
            go.Scatter(
                x=time + time_0,
                y=gt[:, i],
                mode="lines",
                name="GT",
                legendgroup="GT",
                showlegend=(i == 0) and legend,
                line=dict(color=gt_line_color, dash=gt_line_style),
            ),
            row=row,
            col=col,
        )

        # Prediction line (if provided)
        if pred_np is not None:
            fig.add_trace(
                go.Scatter(
                    x=time + time_0,
                    y=pred_np[:, i],
                    mode="lines",
                    name="Pred",
                    legendgroup="Pred",
                    showlegend=(i == 0) and legend,
                    line=dict(color=pred_line_color, dash=pred_line_style),
                ),
                row=row,
                col=col,
            )

        # Quantile area
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time, time[::-1]]) + time_0,
                y=np.concatenate([q_low[:, i], q_high[::-1, i]]),
                fill="toself",
                fillcolor=area_color,
                line=dict(color="rgba(0,0,0,0)"),
                name="CI",
                legendgroup="CI",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        # Outliers
        outliers = (gt[:, i] < q_low[:, i]) | (gt[:, i] > q_high[:, i])
        fig.add_trace(
            go.Scatter(
                x=np.asarray(time[outliers]) + time_0,
                y=gt[outliers, i],
                mode="markers",
                name="Outliers",
                legendgroup="Outliers",
                showlegend=(i == 0),
                marker=dict(color=outlier_color, size=3),
            ),
            row=row,
            col=col,
        )

        # Compute dynamic y-limits
        y_all = gt[:, i]
        y_min = np.min(y_all)
        y_max = np.max(y_all)
        y_margin = (y_max - y_min) * 0.1
        y_lim_min = y_min - y_margin
        y_lim_max = y_max + y_margin
        fig.update_yaxes(range=[y_lim_min, y_lim_max], row=row, col=col)

    # fig.update_layout(height=500 * n_rows, width=1000 * n_cols, title_text=title)
    # Remove borders and white spaces
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
