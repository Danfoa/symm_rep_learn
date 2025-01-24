import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from src.estimators import *
from src.models import *
from src.utils import *


def estimate_mutual_information(
    estimator,
    x,
    y,
    critic_fn,
    baseline_fn=None,
    alpha_logit=None,
    clamping_values=None,
    **kwargs,
):
    """Estimate variational lower bounds on mutual information.

    Args:
      estimator: string specifying estimator, one of:
        'nwj', 'infonce', 'tuba', 'js', 'interpolated', ...
      x: [batch_size, dim_x] Tensor
      y: [batch_size, dim_y] Tensor
      critic_fn: callable that takes x and y as input and outputs critic scores
        output shape is a [batch_size, batch_size] matrix
      baseline_fn (optional): callable that takes y as input
        outputs a [batch_size]  or [batch_size, 1] vector
      alpha_logit (optional): logit(alpha) for interpolated bound

    Returns:
      scalar estimate of mutual information, train_loss
      # note that train_loss may not be MI estimation
    """
    x, y = x.cuda(), y.cuda()
    scores = critic_fn(x, y)
    if clamping_values is not None:
        scores = torch.clamp(scores, min=clamping_values[0], max=clamping_values[1])
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == "nwj":
        return nwj_lower_bound(scores)
    elif estimator == "infonce":
        return infonce_lower_bound(scores)
    elif estimator == "js":
        return js_lower_bound(scores)
    elif estimator == "dv":
        return dv_upper_lower_bound(scores)
    elif estimator == "smile":
        return smile_lower_bound(scores, **kwargs)
    elif estimator == "variational_f_js":
        return variational_f_js(scores)
    elif estimator == "probabilistic_classifier":
        return probabilistic_classifier(scores)
    elif estimator == "density_matching":
        return density_matching(scores)
    elif estimator == "density_matching_lagrange":
        return density_matching_lagrange(scores)
    elif estimator == "density_ratio_fitting":
        return density_ratio_fitting(scores)
    elif estimator == "squared_mutual_information":
        return squared_mutual_information(scores)
    elif estimator == "js_squared_mutual_information":
        return js_squared_mutual_information(scores)


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(p=0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(p=0.1),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)


def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    CRITICS = {"separable": SeparableCritic, "concat": ConcatCritic}

    BASELINES = {
        "constant": lambda: None,
        "unnormalized": lambda: mlp(
            dim=data_params["dim"],
            hidden_dim=512,
            output_dim=1,
            layers=2,
            activation="relu",
        ).cuda(),
    }

    # Ground truth rho is only used by conditional critic
    if mi_params["estimator"] == "ncp":
        from NCP.models.ncp import NCP
        from NCP.mysc.utils import class_from_name

        cfg = OmegaConf.create(
            {
                "batch_size": 128,
                "lr": 1e-3,
                "truncated_op_bias": "full_rank",
                "gamma": 1e-3,
                "embedding": {
                    # They have ~284k params
                    "embedding_dim": 32,
                    "hidden_units": 512,
                    "hidden_layers": 2,
                    "activation": "ReLU",
                },
            }
        )
        print(cfg)
        activation = class_from_name("torch.nn", cfg.embedding.activation)
        kwargs = dict(
            output_shape=cfg.embedding.embedding_dim,
            n_hidden=cfg.embedding.hidden_layers,
            layer_size=cfg.embedding.hidden_units,
            activation=activation,
            bias=False,
            iterative_whitening=False,
        )
        # fx = MLP(input_shape=20, **kwargs)
        # fy = MLP(input_shape=20, **kwargs)
        fx = ImprovedMLP(input_dim=20, hidden_dim=cfg.embedding.hidden_units, output_dim=cfg.embedding.embedding_dim)
        fy = ImprovedMLP(input_dim=20, hidden_dim=cfg.embedding.hidden_units, output_dim=cfg.embedding.embedding_dim)
        critic = NCP(
            embedding_x=fx,
            embedding_y=fy,
            embedding_dim=cfg.embedding.embedding_dim,
            gamma_orthogonality=cfg.gamma,
            truncated_op_bias=cfg.truncated_op_bias,
        ).to("cuda")
    else:
        critic = CRITICS[mi_params.get("critic", "separable")](rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get("baseline", "constant")]()

    opt_crit = optim.Adam(
        critic.parameters(), lr=cfg.lr if mi_params["estimator"] == "ncp" else opt_params["learning_rate"]
    )
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(), lr=opt_params["learning_rate"])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params, trace):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params["critic"] == "conditional":
            critic_ = CRITICS["conditional"](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params["dim"],
            rho=rho,
            batch_size=cfg.batch_size if mi_params["estimator"] == "ncp" else data_params["batch_size"],
            cubic=data_params["cubic"],
        )
        if False:
            mi, p_norm = estimate_mutual_information(
                mi_params["estimator"],
                x,
                y,
                critic_,
                baseline,
                mi_params.get("alpha_logit", None),
                **kwargs,
            )
        if mi_params["estimator"] == "ncp":
            x, y = x.cuda(), y.cuda()
            # ncp_training_step:
            out = critic_(x, y)
            if isinstance(out, tuple):
                loss, metrics = critic_.loss(*out)
            elif isinstance(out, dict):
                loss, metrics = critic_.loss(**out)
            else:
                loss, metrics = critic_.loss(out)

            mi = critic_.pointwise_mutual_information(x, y).mean()
            if trace:
                print(
                    f"MI: {rho_to_mi(20, rho):.1f}; MI_pred: {mi:.1f}; ||k(x,y) - k_r(x,y)||: {metrics['||k(x,y) - k_r(x,y)||']:.1f}; Loss:{loss:.1f}"
                )
            mi, train_obj = mi, loss
        else:
            mi, train_obj = estimate_mutual_information(
                mi_params["estimator"],
                x,
                y,
                critic_,
                baseline,
                mi_params.get("alpha_logit", None),
                **kwargs,
            )
        # The following line is almost surely insanely wrong:
        if mi_params["estimator"] != "ncp":
            loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()

        if False:
            return mi, p_norm
        else:
            return mi, train_obj

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params["iterations"])
    rhos = mi_to_rho(data_params["dim"], mis)

    if False:
        estimates = []
        p_norms = []
        for i in range(opt_params["iterations"]):
            mi, p_norm = train_step(rhos[i], data_params, mi_params)
            mi = mi.detach().cpu().numpy()
            p_norm = p_norm.detach().cpu().numpy()
            estimates.append(mi)
            p_norms.append(p_norm)

        return np.array(estimates), np.array(p_norms)
    else:
        estimates = []
        train_objs = []
        for i in range(opt_params["iterations"]):
            mi, train_obj = train_step(rhos[i], data_params, mi_params, trace=((i % 100) == 0))
            mi = mi.detach().cpu().numpy()
            train_obj = train_obj.detach().cpu().numpy()
            estimates.append(mi)
            train_objs.append(train_obj)

        return np.array(estimates), np.array(train_objs)
