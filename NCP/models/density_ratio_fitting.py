# Created by danfoa at 16/01/25
import torch.nn


class DRF(torch.nn.Module):

    def __init__(self, embedding: torch.nn.Module):
        super(DRF, self).__init__()
        self.embedding = embedding

    # def loss(self, x: torch.Tensor, y: torch.Tensor):
