import numpy as np
import torch
from torch.utils.data import DataLoader

from symm_rep_learn.models.ncp import NCP


class NCPConditionalCDF(torch.nn.Module):
    r"""Estimate the conditional CDF :math:`F_{Y\mid X}(y\mid x)` with a trained NCP.

    This module discretizes the support of :math:`Y` per scalar dimension and learns a
    linear decoder over indicator observables to regress the conditional CDF using the
    conditional expectation operator approximated by the NCP.

    Parameters
    ----------
    model : NCP
        Trained Neural Conditional Process model.
    y_train : torch.Tensor
        Training targets used to define the marginal support, shape ``(n_train, d_y)``.
    support_discretization_points : int, optional
        Number of discretization points of the support grid for each scalar dim of :math:`Y`.
    ridge_reg : float, optional
        Ridge regularization applied when fitting the linear decoder.
    support_strategy : {"linspace", "kbins_discretizer", "quantile_transformer"}, optional
        Strategy to construct a monotone support grid per output dimension. Defaults to "linspace".
    discretizer_kwargs : dict, optional
        Optional keyword arguments specific to the chosen ``support_strategy`` (e.g., ``eps`` for
        "linspace", or parameters for scikit-learn transformers). Robust defaults are used otherwise.

    Notes
    -----
    - The marginal CDF :math:`F_Y` is estimated empirically from indicator functions on the grid.
    - Predicted conditional CDF values are clipped to ``[0, 1]``.
    """

    def __init__(
        self,
        model: NCP,
        y_train,
        support_discretization_points=500,
        ridge_reg=None,
        support_strategy: str = "quantile_transformer",
        discretizer_kwargs: dict | None = None,
    ):
        super(NCPConditionalCDF, self).__init__()
        self.ncp_model = model
        assert y_train.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.ndim}"

        self.discretization_points = support_discretization_points
        self.support_strategy = support_strategy
        self.discretizer_kwargs = discretizer_kwargs or {}

        # Build a robust, monotone support per dimension (shape: (m, d_y)) _____________________________
        self.discretized_support = self._build_support(y_train=y_train, m=self.discretization_points)

        # Indicator functions __________________________________________________________________________
        # Compute the indicator function of (y_i <= y_i') for each y_i' in the discretized support of y_i
        # cdf_obs_ind -> (n_samples, discretization_points, y_dim)
        cdf_obs_ind = (y_train.detach().cpu().numpy()[:, None, :] <= self.discretized_support[None, :, :]).astype(int)

        # Estimate the CDF _____________________________________________________________________________
        self.marginal_CDF = cdf_obs_ind.mean(axis=0)  # (m, d_y) -> F(y') -> P(Y <= y')
        cdf_obs_ind_c = torch.tensor(cdf_obs_ind - self.marginal_CDF, dtype=torch.float32)
        n_samples, _, n_dim = cdf_obs_ind_c.shape
        cdf_obs_ind_c_flat = cdf_obs_ind_c.reshape((n_samples, n_dim * self.discretization_points))
        self.n_obs_dims = y_train.shape[1]

        # Compute the conditional indicator sets per each y' in the support given X=x.
        y_zy_dataloader = DataLoader(
            dataset=torch.utils.data.TensorDataset(y_train, cdf_obs_ind_c_flat),
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )
        self.ccdf_lin_decoder: torch.nn.Linear = model.fit_linear_decoder(
            train_dataloader=y_zy_dataloader, ridge_reg=ridge_reg, lstsq=False
        )
        # Ignore fitted bias
        self.ccdf_lin_decoder.bias = torch.nn.Parameter(torch.zeros(self.ccdf_lin_decoder.bias.shape))

    def _build_support(self, y_train: torch.Tensor, m: int) -> np.ndarray:
        """Construct a monotone support grid of shape ``(m, d_y)`` according to the chosen strategy.

        Ensures each column is non-decreasing. Robust defaults are used to mitigate outliers.
        """
        assert m >= 2, f"support_discretization_points must be >= 2, got {m}"
        y_np = y_train.detach().cpu().numpy()
        n, d = y_np.shape
        support = np.empty((m, d), dtype=float)

        strategy = (self.support_strategy or "linspace").lower()
        kwargs = dict(self.discretizer_kwargs) if self.discretizer_kwargs is not None else {}

        # Defaults per strategy -----------------------------------------------------------------------
        if strategy == "linspace":
            eps = kwargs.get("eps", 0.005)  # winsorization level per tail
            eps = float(np.clip(eps, 0.0, 0.2))
            for j in range(d):
                col = y_np[:, j]
                col_min = np.min(col)
                col_max = np.max(col)
                q_lo = np.quantile(col, eps)
                q_hi = np.quantile(col, 1.0 - eps)
                s = np.linspace(q_lo, q_hi, m)
                # Include original min/max at the ends without changing length
                s[0] = col_min
                s[-1] = col_max
                s = self._enforce_strict_monotone(s)
                support[:, j] = s

        elif strategy == "kbins_discretizer":
            try:
                from sklearn.preprocessing import KBinsDiscretizer
            except Exception as e:
                raise ImportError(
                    "KBinsDiscretizer strategy requires scikit-learn. Install scikit-learn or choose another strategy."
                ) from e
            kbd_kwargs = {
                "n_bins": m,
                "encode": "ordinal",
                "strategy": kwargs.get("strategy", "quantile"),
            }
            kbd_kwargs.update({k: v for k, v in kwargs.items() if k not in {"strategy"}})
            for j in range(d):
                col = y_np[:, j].reshape(-1, 1)
                kbd = KBinsDiscretizer(**kbd_kwargs)
                kbd.fit(col)
                edges = kbd.bin_edges_[0]  # length m+1 in original units
                # Use right edges as thresholds -> m points
                s = edges[1:].astype(float)
                # Handle potential duplicates by falling back to empirical quantiles
                if np.unique(s).size < m:
                    q = np.arange(1, m + 1) / m
                    s = np.quantile(col.ravel(), q)
                s = self._enforce_strict_monotone(s)
                support[:, j] = s

        elif strategy == "quantile_transformer":
            try:
                from sklearn.preprocessing import QuantileTransformer
            except Exception as e:
                raise ImportError(
                    "QuantileTransformer strategy requires scikit-learn. Install scikit-learn or choose another strategy."
                ) from e
            # Prefer uniform so we can grid in [0,1] without extra dependencies
            qt_defaults = {
                "n_quantiles": min(1000, n),
                "output_distribution": kwargs.get("output_distribution", "uniform"),
                "subsample": n,
                "random_state": kwargs.get("random_state", None),
            }
            for j in range(d):
                col = y_np[:, j].reshape(-1, 1)
                qt = QuantileTransformer(**qt_defaults)
                qt.fit(col)
                # Use a clipped uniform grid in [0,1] to avoid extremal instabilities
                u_eps = float(kwargs.get("u_eps", 1e-4))
                u = np.linspace(u_eps, 1.0 - u_eps, m)
                if qt_defaults["output_distribution"] == "uniform":
                    s = qt.inverse_transform(u.reshape(-1, 1)).ravel()
                else:
                    raise NotImplementedError(
                        f"Output distribution '{qt_defaults['output_distribution']}' not supported."
                    )
                s = self._enforce_strict_monotone(s)
                support[:, j] = s

        else:
            raise ValueError(
                f"Unknown support_strategy '{self.support_strategy}'. Choose from 'linspace', 'kbins_discretizer', 'quantile_transformer'."
            )

        return support

    def _enforce_strict_monotone(self, s: np.ndarray) -> np.ndarray:
        """Make sequence strictly increasing with a tiny minimal step to avoid zero-length intervals.

        The minimal step is adaptive to the span and grid size; if the span is near-zero,
        a very small absolute epsilon is used.
        """
        s = np.asarray(s, dtype=float).copy()
        s = np.maximum.accumulate(s)
        span = float(s[-1] - s[0])
        # Adaptive minimal step; can be overridden via discretizer_kwargs["min_step"]
        min_step_cfg = (self.discretizer_kwargs or {}).get("min_step", None)
        if min_step_cfg is None:
            min_step = max(1e-12, span / max(10_000, 10 * max(1, len(s) - 1)))
        else:
            min_step = float(min_step_cfg)
        for i in range(1, len(s)):
            if s[i] <= s[i - 1]:
                s[i] = s[i - 1] + min_step
        return s

    def forward(self, x_cond: torch.Tensor):
        r"""Predict :math:`F_{Y\mid X}(y'\mid x)` on the predefined support grid.

        Parameters
        ----------
        x_cond : torch.Tensor
            Conditioning inputs :math:`X`, shape ``(n, d_x)`` or ``(d_x,)``.

        Returns
        -------
        numpy.ndarray
            Conditional CDF array of shape ``(n, m, d_y)`` where ``m`` is the number of
            support points (or ``(m, d_y)`` when ``n=1``).
        """
        device = next(self.ncp_model.parameters()).device
        self.to(device)

        if x_cond.ndim == 1:
            x_cond = x_cond[None, :]

        deflated_ccdf_pred = self.ncp_model.conditional_expectation(x_cond.to(device), hy2zy=self.ccdf_lin_decoder)
        deflated_ccdf_pred = deflated_ccdf_pred.detach().cpu().numpy()  # (n_cond_points, discretization_points * y_dim)
        # print(deflated_ccdf_pred.shape)
        n_cond_points, _ = deflated_ccdf_pred.shape
        deflated_ccdf_pred = deflated_ccdf_pred.reshape((n_cond_points, self.discretization_points, self.n_obs_dims))
        ccdf_pred = self.marginal_CDF + deflated_ccdf_pred
        # Smooth the predicted Conditional Cumulative Distribution Function
        # TODO: Implement smoothing for multivariate CDF
        # Filter out points below 0 from approximation of the NCP [0, 1]
        ccdf_pred = np.clip(ccdf_pred, 0, 1)

        assert ccdf_pred.shape == (n_cond_points, self.discretization_points, self.n_obs_dims)
        ccdf_pred = np.squeeze(ccdf_pred, axis=0) if n_cond_points == 1 else ccdf_pred
        return ccdf_pred

    def conditional_quantiles(self, x_cond: torch.Tensor, alpha=0.05):
        r"""Compute per-dimension shortest intervals at coverage :math:`1-\alpha`.

        For each dimension :math:`i`, returns :math:`[q^{(i)}_{\text{low}}, q^{(i)}_{\text{high}}]` on the
        grid such that :math:`F_{Y_i\mid X}(q^{(i)}_{\text{high}}\mid x) - F_{Y_i\mid X}(q^{(i)}_{\text{low}}\mid x) \ge 1-\alpha`.

        Parameters
        ----------
        x_cond : torch.Tensor
            Conditioning inputs :math:`X`, shape ``(n, d_x)`` or ``(d_x,)``.
        alpha : float, optional
            Miscoverage level :math:`\alpha \in (0, 1]`.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Arrays ``(q_low, q_high)`` each of shape ``(n, d_y)`` (or ``(d_y,)`` when ``n=1``).
        """
        assert alpha is None or 0 < alpha <= 1, f"Alpha must be in the range (0, 1] got {alpha}"
        ccdf = self.forward(x_cond)
        q_low, q_high = [], []
        for dim in range(self.n_obs_dims):
            dim_ccdf = ccdf[..., dim]  # (n_train_points, discretization_points)
            if dim_ccdf.ndim == 2:  # Multiple conditioning points:
                q_low_per_x, q_high_per_ = [], []
                for x_cond_idx in range(dim_ccdf.shape[0]):
                    low_qx, high_qx = self.find_best_quantile(
                        self.discretized_support[..., dim], dim_ccdf[x_cond_idx], alpha
                    )
                    q_low_per_x.append(low_qx)
                    q_high_per_.append(high_qx)
                q_low.append(np.asarray(q_low_per_x))
                q_high.append(np.asarray(q_high_per_))
            elif dim_ccdf.ndim == 1:
                low_qx, high_qx = self.find_best_quantile(self.discretized_support[..., dim], dim_ccdf, alpha)
                q_low.append(low_qx)
                q_high.append(high_qx)
            else:
                raise ValueError(f"Invalid shape {dim_ccdf.shape}")
        q_low = np.asarray(q_low).T
        q_high = np.asarray(q_high).T
        return q_low, q_high

    @staticmethod
    def find_best_quantile(x_support, cdf_x, alpha):
        r"""Find the shortest grid interval with mass :math:`\ge 1-\alpha`.

        Given increasing support values and their CDF values, returns the pair
        :math:`(q_{\text{low}}, q_{\text{high}})` minimizing interval length under the
        constraint :math:`F(q_{\text{high}}) - F(q_{\text{low}}) \ge 1-\alpha`.
        """
        # Ensure 1D arrays
        x_support = np.asarray(x_support).flatten()
        cdf_x = np.asarray(cdf_x).flatten()
        m = len(cdf_x)
        if m < 2:
            return x_support[0], x_support[-1]
        # Enforce monotonicity defensively
        x_support = np.maximum.accumulate(x_support)
        # cdf_x = np.clip(np.maximum.accumulate(cdf_x), 0.0, 1.0)

        t0 = 0
        t1 = 1
        best_t0 = 0
        best_t1 = m - 1
        best_size = x_support[best_t1] - x_support[best_t0]
        while t0 < m:
            # stop if left border reaches right end of discretisation
            if cdf_x[t1] - cdf_x[t0] >= 1 - alpha:
                # if x[t0], x[t1] is a confidence interval at level alpha, compute length and compare to best
                size = x_support[t1] - x_support[t0]
                if size < best_size:
                    best_t0 = t0
                    best_t1 = t1
                    best_size = size
                # moving t1 to the right will only increase the size of the interval, so we can safely move t0 to the right
                t0 += 1
                if t0 >= t1 and t1 < m - 1:
                    t1 = t0 + 1
            elif t1 == m - 1:
                # no more pertinent intervals
                break
            else:
                # increase right edge
                t1 += 1
        return x_support[best_t0], x_support[best_t1]
