import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
import random
from typing import Union


class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=1, activ=f.relu, output_nonlinearity=None, n_units=64):
        super(MlpNetwork, self).__init__()
        # n_units = 512
        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
        self.out_nl = output_nonlinearity
        self.activ = activ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of network
        :param x:
        :return:
        """
        x = self.activ(self.h1(x))
        x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == f.log_softmax:
                x = f.log_softmax(x, dim=-1)
            else:
                x = self.out_nl(x)
        return x


class Predictor(nn.Module):
    def __init__(self, x_dim=1, reward_type='aim'):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.set_default_dtype(torch.float32)
        super(Predictor, self).__init__()
        self.input_dim = x_dim
        self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)
        self.d.to(self.device)
        self.n = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        return discriminator output
        :param x:
        :return:
        """
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        if self.use_cuda:
            x = x.cuda()
        output = self.d(x)
        return output

    def reward(self, x: np.ndarray) -> np.ndarray:
        """
        return the reward
        """
        # x = x.astype(np.float64)
        r = - self.forward(x)
        if self.n < 99:
            r *= 0.
        return r.cpu().detach().numpy()

    def update(self, inputs: np.ndarray,
               dist: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        self.n = min(self.n + 1, 100)
        self.optimizer.zero_grad()
        dist = torch.from_numpy(dist)
        masks = torch.from_numpy(masks)
        if self.use_cuda:
            dist = dist.cuda()
            masks = masks.cuda()
        preds = self(inputs)
        error = torch.square(preds - dist)
        masked_error = masks * error
        loss = torch.mean(masked_error)

        loss.backward()
        # utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.optimizer.step()
        return loss.cpu().detach().numpy()
