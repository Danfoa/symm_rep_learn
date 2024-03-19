import torch

from NCP.nn.functional import cme_score
from NCP.layers import SingularLayer

class CMELoss():
    def __init__(
        self, gamma:float
    ):
        """Initializes the CME loss of CITATION NEEDED

        Args:
            gamma (float): penalisation parameter.
        """
        self.gamma = gamma


    def __call__(self, X1: torch.Tensor, X2:torch.Tensor, Y1: torch.Tensor, Y2: torch.Tensor, S: SingularLayer):
        """Compute the Deep Projection loss function

        Args:
            X1 (torch.Tensor): Covariates for the initial time steps.
            X2 (torch.Tensor): .
            Y1 (torch.Tensor): Covariates for the evolved time steps.
            Y2 (torch.Tensor): .
            S (SingularLayer): .
        """
        return cme_score(X1, X2, Y1, Y2, S, self.gamma)