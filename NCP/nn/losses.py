import torch
from NCP.nn.functional import cme_score_cov, cme_score_Ustat, cme_score_opti
from NCP.nn.layers import SingularLayer
from NCP.model import NCPOperator
from NCP.nn.diffusion_conditional import DDPM
from normflows import MultiscaleFlow

class CMELoss():
    def __init__(
            self,
            mode: str = "split",
            gamma:float = 0.,
            metric_deformation: float = 1.0,
            center: bool = True
    ):
        """Initializes the CME loss of CITATION NEEDED

        Args:
            gamma (float): penalisation parameter.
        """
        available_modes = ["split", "cov", "U_stat"]
        if mode not in available_modes:
            raise ValueError(f"Unknown mode {mode}. Available modes are {available_modes}")
        else:
            self.mode = mode

        if self.mode in ["split", "cov"]:
            self.gamma = gamma
        else:
            self.metric_deformation = metric_deformation
            self.center = center

    def __call__(self, X: torch.Tensor, Y:torch.Tensor, NCP: NCPOperator):
        """Compute the Deep Projection loss function

        Args:
            X (torch.Tensor): Covariates for the initial time steps.
            Y (torch.Tensor): Covariates for the evolved time steps.
            S (SingularLayer): .
        """
        if self.mode == "split":
            return cme_score_opti(X, Y, NCP, self.gamma)
        if self.mode == "cov":
            return cme_score_cov(X, Y, NCP, self.gamma)
        else:
            return cme_score_Ustat(X, Y, NCP, self.metric_deformation, self.center)
        
class DDPMLoss():
    def __init__(self,
            mode: str = "split",
            gamma:float = 0.,
            metric_deformation: float = 1.0,
            center: bool = True):
        pass

    def __call__(self, X:torch.Tensor, Y:torch.Tensor, ddpm: DDPM):
        return ddpm(X, Y)
    
class NFLoss():
    def __init__(self,
            mode: str = "split",
            gamma:float = 0.,
            metric_deformation: float = 1.0,
            center: bool = True):
        pass
    
    def __call__(self, X:torch.Tensor, Y:torch.Tensor, model: MultiscaleFlow):
        return model.forward_kld(Y, X)


class KMNLoss():
    def __init__(self,
                 mode: str = "split",
                 gamma: float = 0.,
                 metric_deformation: float = 1.0,
                 center: bool = True):
        pass

    def __call__(self, X, Y, model):
        out = model(X)
        weights = out[:, 0:-1]
        sigma = torch.unsqueeze(out[:, -1], dim=1)
        return model.mdn_loss_function(weights, sigma, Y)