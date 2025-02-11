# Created by danfoa at 16/01/25
import torch.nn

from symm_rep_learn.nn.layers import Lambda


# Density Ratio Fitting.
class DRF(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Module, gamma: float = 0.01):
        super(DRF, self).__init__()
        self.gamma = gamma
        self.pmd = torch.nn.Sequential(
            embedding,
            Lambda(lambda x: 1 + x),  # 1 of conditional independence, NN learns the remainder part.
            torch.nn.Softplus(),  # Ensure estimates of PMD are positive and that gradients are well-behaved
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes the estimations of the Pointwise Mutual Dependency ratio for each pair of x and y.

        Args:
            x: (n_samples, x_dim) tensor
            y: (n_samples, y_dim) tensor
        Returns:
            PMD: (n_samples, n_samples) tensor containing the PMD ratio for all pairwise combinations of x and y.
                Where the diagonal elements are the PMD ratio of (x_i,y_i) ~ p(x,y) and the off-diagonal elements are
                the PMD ratio of (x_i,y_j) ~ p(x)p(y).
        """
        n_samples, x_dim = x.shape
        _, y_dim = y.shape

        # Repeat x and y to create all combinations
        x_repeated = x.repeat_interleave(n_samples, dim=0)
        y_repeated = y.repeat(n_samples, 1)

        # Concatenate x and y to form the xy pairs
        xy_pairs = torch.cat((x_repeated, y_repeated), dim=1)

        # Forward pass through the embedding function
        PMD = self.pmd(xy_pairs)
        assert PMD.shape == (n_samples**2, 1)

        # Reshape the scores back to (n_samples, n_samples)
        PMD_mat = PMD.view(n_samples, n_samples)

        return PMD_mat

    def pointwise_mutual_dependency(self, x: torch.Tensor, y: torch.Tensor):
        xy = torch.cat((x, y), dim=1)
        return self.pmd(xy).squeeze()

    def pointwise_mutual_information(self, x: torch.Tensor, y: torch.Tensor):
        PMD = self.pointwise_mutual_dependency(x, y)
        PMD_pos = torch.clamp(PMD, min=1e-6)  # Need to clamp to avoid NaNs.
        pmi = torch.log(PMD_pos)
        assert torch.isfinite(pmi).all(), "NaN or Inf values found in the PMI estimation"
        return pmi

    def loss(self, pmd_mat: torch.Tensor):
        """Computes the Density Ratio Fitting Loss.

        Args:
            pmd_mat: (n_samples, n_samples) tensor containing the PMD ratio for all pairwise combinations of x and y.

        Returns:
            loss: (1,) tensor computed as E_p(x,y)[pmd(x,y)] - E_p(x)p(y)[pmd(x,y)^2]
        """
        assert pmd_mat.shape[0] == pmd_mat.shape[1], f"Expected (n_samples, n_samples) tensor, got {pmd_mat.shape}"

        n_samples = pmd_mat.shape[0]
        # E_p(x,y)[pmd(x,y)]
        E_pxy = torch.diag(pmd_mat).mean()
        pmd_2 = pmd_mat**2
        # E_p(x)p(y)[pmd(x,y)^2]
        E_px_py = (pmd_2.sum() - pmd_2.diag().sum()) / (n_samples * (n_samples - 1))

        # Penalization term: 1 - E_p(x)p(y)[pmd(x,y)]
        prob_mass_penalization = (1 - pmd_mat.mean()) ** 2

        # L(x,y) = -2 E_p(x,y)[pmd(x,y)] + E_p(x)p(y)[pmd(x,y)^2] + 1 - λ * ||E_p(x)p(y)[pmd(x,y)]||^2
        density_ratio_err = (-2 * E_pxy) + (E_px_py) + 1  # Deflated loss
        loss = density_ratio_err + self.gamma * prob_mass_penalization
        metrics = {
            "||k(x,y) - k_r(x,y)||": density_ratio_err.detach(),
            "E_p(x)p(y) k_r(x,y)^2": E_px_py.detach() - 1,
            "E_p(x,y) k_r(x,y)": E_pxy.detach() - 1,
            "Prob Mass Distortion": prob_mass_penalization.detach(),
        }
        return loss, metrics

    def extra_repr(self) -> str:
        return "Density Ratio Fitting"


if __name__ == "__main__":
    from symm_rep_learn.nn.layers import MLP, Lambda

    in_dim = 10

    fxy = MLP(
        input_shape=in_dim,
        output_shape=1,
        n_hidden=3,
        layer_size=64,
        activation=torch.nn.GELU,
    )
    drf = DRF(embedding=fxy)

    x = torch.randn(10, in_dim // 2)
    y = torch.randn(10, in_dim // 2)

    scores = drf(x, y)

    loss = drf.loss(scores)
    # Compute individual scores for each pair (x_i, y_i)
    joint_scores = []
    for i in range(x.shape[0]):
        xy_pair = torch.cat((x[i].unsqueeze(0), y[i].unsqueeze(0)), dim=1)
        individual_score = drf.pmd(xy_pair)
        joint_scores.append(individual_score.item())

    # Extract the diagonal elements from the scores matrix
    joint_scores_diag = torch.diag(scores.squeeze())

    # Check if the individual scores match the diagonal elements
    assert torch.allclose(
        torch.tensor(joint_scores), joint_scores_diag, rtol=1e-5, atol=1e-5
    ), f"err_max = {torch.max(torch.abs(torch.tensor(joint_scores) - joint_scores_diag))}"

    print("Test passed: Diagonal scores match individual scores")
