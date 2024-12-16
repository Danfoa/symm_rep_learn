
import numpy as np
import torch
import torch.nn as nn

# code from https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
#copyright 2022 Tim Pearce (MIT License)

class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualBlock(in_channels, out_channels)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.Linear(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(ContextUnet, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.init_conv = ResidualBlock(y_dim, y_dim, is_res=True)

        self.down1 = UnetDown(y_dim, y_dim)
        self.down2 = UnetDown(y_dim, 2 * y_dim)

        self.timeembed1 = EmbedFC(1, 2*y_dim)
        self.timeembed2 = EmbedFC(1, 1*y_dim)
        self.contextembed1 = EmbedFC(x_dim, 2*y_dim)
        self.contextembed2 = EmbedFC(x_dim, 1*y_dim)

        self.up0 = nn.Sequential(
            nn.Linear(2 * y_dim, 2 * y_dim),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * y_dim, y_dim)
        self.up2 = UnetUp(2 * y_dim, y_dim)
        self.out = nn.Sequential(
            nn.Linear(2 * y_dim, y_dim),
            nn.ReLU(),
            nn.Linear(y_dim, self.y_dim),
        )

    def forward(self, x, c, t):
        # x is (noisy) image, c is context label, t is timestep
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        # embed context, time step
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t[:,None])

        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t[:,None])

        up1 = self.up0(down2)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings

        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, y):
        """This method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (y.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(y)  # eps ~ N(0, 1)

        y_t = (
            self.sqrtab[_ts, None] * y
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(y_t, x, _ts / self.n_T))

    def sample(self, context, n_sample, device, guide_w = 0.):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance


        y_i = torch.randn(n_sample, self.nn_model.y_dim).to(device)  # x_T ~ N(0, 1), sample initial noise
        x_i = context.repeat(n_sample, 1).to(device)
        # double the batch
        x_i = x_i.repeat(2, 1)

        y_i_store = [] # keep track of generated steps in case want to plot something
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            # double batch
            y_i = y_i.repeat(2,1)
            t_is = t_is.repeat(2)

            z = torch.randn(n_sample, self.nn_model.y_dim).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_i, t_is)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            y_i = y_i[:n_sample]
            y_i = (
                self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                y_i_store.append(y_i.detach().cpu().numpy())

        y_i_store = np.array(y_i_store)
        return y_i.detach().cpu().numpy(), y_i_store

