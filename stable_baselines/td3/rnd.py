import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
from typing import Union
from stable_baselines.common.running_mean_std import RunningMeanStd


class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=64, activ=f.relu, n_units=64):
        super(MlpNetwork, self).__init__()
        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
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
        return x


class RND(nn.Module):
    def __init__(self, x_dim=1, embedding_dim=32):
        self.use_cuda = torch.cuda.is_available()
        # gpu = 1
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.set_default_dtype(torch.float32)
        super(RND, self).__init__()
        self.input_dim = x_dim
        self.e_dim = embedding_dim

        self.target = MlpNetwork(self.input_dim, n_units=32, output_dim=self.e_dim)  # , activ=f.tanh)
        self.target.to(self.device)
        self.learner = MlpNetwork(self.input_dim, n_units=32, output_dim=self.e_dim)  # , activ=f.tanh)
        self.learner.to(self.device)
        self.obs_stats = RunningMeanStd(shape=(self.input_dim,))
        self.rew_stats = RunningMeanStd()
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=1e-3)
        self.n = 0

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        return discriminator output
        :param x:
        :return:
        """
        x = np.clip((x - self.obs_stats.mean) / np.sqrt(self.obs_stats.var), -5., 5.)
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        if self.use_cuda:
            x = x.cuda()
        t_emb = self.target(x)
        f_emb = self.learner(x)
        mse = torch.mean(torch.square(t_emb.detach() - f_emb), dim=-1)
        return mse

    def reward(self, x: np.ndarray) -> np.ndarray:
        """
        return the reward
        """
        # x = x.astype(np.float64)
        mse = self.forward(x).cpu().detach().numpy()
        r = mse / np.sqrt(self.rew_stats.var)
        if self.n < 99:
            return np.zeros_like(r)
        return r

    def update(self, states: np.ndarray) -> np.ndarray:
        """
        Optimize the learner based on the target outputs
        :return:
        """
        self.optimizer.zero_grad()
        self.obs_stats.update(states)
        mse = self(states)

        loss = torch.mean(mse)
        loss.backward()
        self.optimizer.step()
        self.rew_stats.update(mse.cpu().detach().numpy())
        self.n = min(self.n + 1, 100)
        return loss.cpu().detach().numpy()
