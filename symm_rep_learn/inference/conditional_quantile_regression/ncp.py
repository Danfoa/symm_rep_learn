import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from symm_rep_learn.models import NCP

log = logging.getLogger(__name__)


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
        support_discretization_points=100,
        ridge_reg=None,
        support_strategy: str = "quantile_transformer",
        discretizer_kwargs: dict | None = None,
    ):
        super(NCPConditionalCDF, self).__init__()
        self.ncp_model = model
        assert y_train.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.shape}"

        self.discretization_points = support_discretization_points
        self.support_strategy = support_strategy
        self.discretizer_kwargs = discretizer_kwargs or {}

        # Build a robust, monotone support per dimension (shape: (m, d_y)) _____________________________
        log.info("Discretizing support for conditional CDF estimation...")
        self.discretized_support = self._build_support(
            y_train=y_train, n_discretization_points=self.discretization_points
        )
        log.info(f"Discretized support shape: {self.discretized_support.shape}")

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

        self.ccdf_lin_decoder = self._fit_ccdf_linear_decoder(y_train, cdf_obs_ind_c_flat)

        log.info("NCP Conditional CDF model initialized.")

    def _fit_ccdf_linear_decoder(self, y_train, cdf_obs_ind_c_flat: torch.Tensor) -> torch.nn.Linear:
        # Compute the conditional indicator sets per each y' in the support given X=x.
        y_zy_dataloader = DataLoader(
            dataset=torch.utils.data.TensorDataset(y_train, cdf_obs_ind_c_flat),
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )
        ccdf_lin_decoder = self.ncp_model.fit_linear_decoder(train_dataloader=y_zy_dataloader)
        # Ignore fitted bias
        ccdf_lin_decoder.bias = torch.nn.Parameter(torch.zeros(ccdf_lin_decoder.bias.shape))

        return ccdf_lin_decoder

    def _build_support(self, y_train: torch.Tensor, n_discretization_points: int) -> np.ndarray:
        """Construct a monotone support grid of shape ``(m, d_y)`` according to the chosen strategy.

        Ensures each column is non-decreasing. Robust defaults are used to mitigate outliers.
        """
        assert n_discretization_points >= 2, (
            f"support_discretization_points must be >= 2, got {n_discretization_points}"
        )
        y_np = y_train.detach().cpu().numpy()
        n, d = y_np.shape
        support = np.empty((n_discretization_points, d), dtype=float)

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
                s = np.linspace(q_lo, q_hi, n_discretization_points)
                # Include original min/max at the ends without changing length
                s[0] = col_min
                s[-1] = col_max
                s = self._enforce_strict_monotone(s)
                support[:, j] = s

        elif strategy == "kbins_discretizer":
            from sklearn.preprocessing import KBinsDiscretizer

            kbd_kwargs = {
                "n_bins": n_discretization_points,
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
                if np.unique(s).size < n_discretization_points:
                    q = np.arange(1, n_discretization_points + 1) / n_discretization_points
                    s = np.quantile(col.ravel(), q)
                s = self._enforce_strict_monotone(s)
                support[:, j] = s

        elif strategy == "quantile_transformer":
            from sklearn.preprocessing import QuantileTransformer

            out_distribution = kwargs.get("output_distribution", "normal")
            qt_defaults = dict(
                n_quantiles=n_discretization_points,
                output_distribution=out_distribution,
                subsample=n,  # use all data
                random_state=kwargs.get("random_state", None),
            )
            # Fit once on the full (n, d) data matrix; features are transformed independently
            qt = QuantileTransformer(**qt_defaults)
            qt.fit(y_np)
            if out_distribution == "uniform":
                # Grid uniformly in (0,1) with clipping to avoid extremal instabilities
                u_eps = float(kwargs.get("u_eps", 1e-5))
                u = np.linspace(u_eps, 1.0 - u_eps, n_discretization_points)
                U = np.tile(u.reshape(-1, 1), (1, d))  # (m, d)
                S = qt.inverse_transform(U)  # (m, d) in original units per feature
            elif out_distribution == "normal":
                # Grid linearly in normal space within [-n_std, n_std], then invert to original units
                n_std = float(kwargs.get("n_std", 3.0))
                z = np.linspace(-n_std, n_std, n_discretization_points)
                Z = np.tile(z.reshape(-1, 1), (1, d))  # (m, d)
                S = qt.inverse_transform(Z)  # (m, d)
            else:
                raise NotImplementedError(
                    f"Output distribution '{out_distribution}' not supported. Use 'uniform' or 'normal'."
                )

            # Enforce strict monotonicity per feature
            for j in range(d):
                support[:, j] = self._enforce_strict_monotone(S[:, j])

        else:
            raise ValueError(
                f"Unknown support_strategy '{self.support_strategy}'. Choose from 'linspace', 'kbins_discretizer', 'quantile_transformer'."
            )

        # Final sanity check: ensure we return exactly m points per dimension
        assert support.shape == (n_discretization_points, d), (
            f"Built support has shape {support.shape}, expected {(n_discretization_points, d)}"
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
        # TODO: Implement smoothing of the estimated cCDFs
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
        # Vectorized path -----------------------------------------------------------------------------
        if ccdf.ndim == 3:
            q_low, q_high = self.find_best_quantile_batched(self.discretized_support, ccdf, alpha)
            return q_low, q_high
        elif ccdf.ndim == 2:
            # Single conditioning point; still use batched for consistency
            q_low_b, q_high_b = self.find_best_quantile_batched(self.discretized_support, ccdf[None, ...], alpha)
            return q_low_b[0], q_high_b[0]
        else:
            raise ValueError(f"Invalid ccdf shape {ccdf.shape}")

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

    @staticmethod
    def find_best_quantile_batched(x_support: np.ndarray, cdf: np.ndarray, alpha: float):
        r"""Vectorized shortest-interval quantiles for many conditioning points and dimensions.

        Args:
            x_support: array of shape (m, d)
            cdf: array of shape (n, m, d) or (m, d)
            alpha: miscoverage level in (0,1]

        Returns:
            (q_low, q_high): arrays of shape (n, d) (or squeezed to (d,) if n==1 when caller handles it).
        """
        assert 0 < alpha <= 1, f"alpha must be in (0,1], got {alpha}"
        x_support = np.asarray(x_support)
        cdf = np.asarray(cdf)
        assert x_support.ndim == 2, f"x_support must be (m,d), got {x_support.shape}"
        if cdf.ndim == 2:
            cdf = cdf[None, ...]
        assert cdf.ndim == 3, f"cdf must be (n,m,d), got {cdf.shape}"
        n, m, d = cdf.shape
        assert x_support.shape == (m, d), f"support shape {x_support.shape} != {(m, d)}"

        # Torch implementation using batched searchsorted along m -------------------------------------
        C = torch.from_numpy(cdf.astype(np.float32))  # (n,m,d)
        S = torch.from_numpy(x_support.astype(np.float32))  # (m,d)
        p = 1.0 - float(alpha)
        # Permute to (n,d,m) to treat last dim as search axis
        C_ndm = C.permute(0, 2, 1).contiguous()  # (n,d,m)
        T = (C_ndm + p).clamp(max=1.0)  # (n,d,m)
        # searchsorted requires sorted input along last dim; assume ccdf is monotone along m
        idx_raw = torch.searchsorted(C_ndm, T, right=False)  # (n,d,m), values in [0, m]
        idx_clamped = idx_raw.clamp(max=m - 1)

        # Feasibility: C[idx] - C[t0] >= p and idx < m
        C_right = torch.gather(C_ndm, dim=2, index=idx_clamped)
        C_left = C_ndm  # broadcast over t0 axis
        feasible = (idx_raw < m) & ((C_right - C_left) >= (p - 1e-12))

        # Widths: xR - xL
        S_T = S.t()  # (d,m)
        S_T_exp = S_T.unsqueeze(0).expand(n, -1, -1)  # (n,d,m)
        xR = torch.gather(S_T_exp, dim=2, index=idx_clamped)  # (n,d,m)
        xL = S_T_exp  # (n,d,m) where column t0 is left point
        widths = xR - xL  # (n,d,m)
        widths[~feasible] = float("inf")

        # Argmin over left index t0
        best_t0 = torch.argmin(widths, dim=2)  # (n,d)
        best_t1 = torch.gather(idx_clamped, 2, best_t0.unsqueeze(-1)).squeeze(-1)  # (n,d)

        # Fallback for cases with no feasible interval: choose full span [0, m-1]
        any_feasible = feasible.any(dim=2)  # (n,d)
        if not torch.all(any_feasible):
            default_t0 = torch.zeros_like(best_t0)
            default_t1 = (m - 1) * torch.ones_like(best_t1)
            best_t0 = torch.where(any_feasible, best_t0, default_t0)
            best_t1 = torch.where(any_feasible, best_t1, default_t1)

        q_low = torch.gather(S_T_exp, 2, best_t0.unsqueeze(-1)).squeeze(-1)  # (n,d)
        q_high = torch.gather(S_T_exp, 2, best_t1.unsqueeze(-1)).squeeze(-1)  # (n,d)

        return q_low.cpu().numpy(), q_high.cpu().numpy()


if __name__ == "__main__":
    # Minimal tests: vectorized vs non-vectorized agree for multiple conditioning points and dims
    rng = np.random.default_rng(0)
    n, m, d = 7, 57, 3
    # Build strictly increasing support per dim
    S = np.sort(rng.normal(size=(m, d)), axis=0)
    # Ensure strictly increasing by adding tiny steps if needed
    for j in range(d):
        S[:, j] = np.maximum.accumulate(S[:, j])
        for i in range(1, m):
            if S[i, j] <= S[i - 1, j]:
                S[i, j] = S[i - 1, j] + 1e-8
    # Build monotone CDFs in [0,1]
    raw = rng.random(size=(n, m, d))
    raw.sort(axis=1)
    C = np.clip(raw, 0.0, 1.0)

    alpha = 0.2
    # Vectorized
    ql_v, qh_v = NCPConditionalCDF.find_best_quantile_batched(S, C, alpha)

    # Non-vectorized baseline
    ql_b = np.zeros((n, d))
    qh_b = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            ql_b[i, j], qh_b[i, j] = NCPConditionalCDF.find_best_quantile(S[:, j], C[i, :, j], alpha)

    # Compare
    assert np.allclose(ql_v, ql_b, atol=1e-6), f"q_low mismatch\nvec={ql_v}\nbase={ql_b}"
    assert np.allclose(qh_v, qh_b, atol=1e-6), f"q_high mismatch\nvec={qh_v}\nbase={qh_b}"
    print("Vectorized and baseline quantiles match for all test cases.")
