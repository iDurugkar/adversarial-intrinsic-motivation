import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
import random
from typing import Union


def wasserstein_reward(d: torch.Tensor) -> torch.Tensor:
    """
    return the wasserstein reward
    """
    return d


def gail_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminaotr output and return the gail reward
    :param d:
    :return:
    """
    d = torch.sigmoid(d)
    return d.log()  # - (1 - d).log()


def airl_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminaotr output and return AIRL reward
    :param d:
    :return:
    """
    s = torch.sigmoid(d)
    reward = s.log() - (1 - s).log()
    return reward


def fairl_reward(d: torch.Tensor) -> torch.Tensor:
    """
    Take discriminator output and return FAIRL reward
    :param d:
    :return:
    """

    d = torch.sigmoid(d)
    h = d.log() - (1 - d).log()
    h = torch.clamp(h, -10., 10.)
    return h.exp() * (-h)

reward_mapping = {'aim': wasserstein_reward,
                  'gail': gail_reward,
                  'airl': airl_reward,
                  'fairl': fairl_reward}


class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=1, activ=f.relu, output_nonlinearity=None, n_units=64):
        super(MlpNetwork, self).__init__()
        # n_units = 512
        self.h1 = nn.Linear(input_dim, n_units)
        # self.h2 = nn.Linear(n_units, n_units)
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
        # x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == f.log_softmax:
                x = f.log_softmax(x, dim=-1)
            else:
                x = self.out_nl(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, x_dim=1, reward_type='aim'):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.set_default_dtype(torch.float32)
        super(Discriminator, self).__init__()
        self.input_dim = x_dim
        assert reward_type in ['aim', 'gail', 'airl', 'fairl']
        # assert reward_type in [wasserstein_reward, gail_reward, airl_reward, fairl_reward]
        self.reward_type = reward_mapping[reward_type]
        if self.reward_type == 'aim':
            self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)
        else:
            self.d = MlpNetwork(self.input_dim, n_units=64, activ=f.tanh)
        self.d.to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

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
        r = self.forward(x)
        if self.reward_type is not None:
            r = self.reward_type(r)
        return r.cpu().detach().numpy()

    def compute_graph_pen(self,
                          prev_state: torch.Tensor,
                          next_state_state: torch.Tensor, lambda_=10.):
        """
        Computes values of the discriminator at different points
        and constraints the difference to be 0.1
        """
        # targ_next_state = torch.clone(target_state)
        # idx = targ_next_state.size(0)
        # targ_next_state[0:idx//3, 0] += 1.
        # targ_next_state[idx // 3: 2 * idx//3, 0] -= 1.
        # targ_next_state[2 * idx // 3:, 1] += 1.

        # prevs = torch.cat([target_state, prev_state], dim=0)
        # nexts = torch.cat([targ_next_state, next_state_state], dim=0)
        if self.use_cuda:
            prev_state = prev_state.cuda()
            next_state_state = next_state_state.cuda()
            zero = torch.zeros(size=[int(next_state_state.size(0))]).cuda()
        else:
            zero = torch.zeros(size=[int(next_state_state.size(0))])
        prev_out = self(prev_state)
        next_out = self(next_state_state)
        penalty = lambda_ * torch.max(torch.abs(next_out - prev_out) - 0.1, zero).pow(2).mean()
        return penalty

    def compute_grad_pen(self,
                         target_state: torch.Tensor,
                         policy_state: torch.Tensor,
                         lambda_=10.):
        """
        Computes the gradients by mixing the data randomly
        and creates a loss for the magnitude of the gradients.
        """
        if self.use_cuda:
            target_state = target_state.cuda()
            policy_state = policy_state.cuda()
        alpha = torch.rand(target_state.size(0), 1)
        # expert_data = torch.cat([expert_state, expert_action], dim=1)
        # policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(target_state).to(target_state.device)

        mixup_data = alpha * target_state + (1 - alpha) * policy_state
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def optimize_discriminator(self, target_states: np.ndarray, policy_states:  np.ndarray,
                               policy_next_states: np.ndarray) -> np.ndarray:
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        # num_samples = 50
        # target_states = target_states.astype(np.float64)
        # policy_states = policy_states.astype(np.float64)
        self.discriminator_optimizer.zero_grad()
        # _, _, _, target_distribution, _ = self.target_buffer.sample(100)
        # target_dist = np.reshape(self.env.target_distribution(), (-1,))
        # target_distribution = np.random.choice(target_dist.shape[0], num_samples, p=target_dist)
        ones = target_states
        zeros = policy_next_states
        zeros_prev = policy_states

        # ########## GAIL loss
        if self.reward_type != wasserstein_reward:
            num_samples = ones.shape[0]
            labels_ones = torch.ones((num_samples, 1)) * 0.9
            labels_zeros = torch.ones((num_samples, 1)) * 0.1
            data = np.concatenate([ones, zeros])
            pred = self(data)
            labels = torch.cat([labels_ones, labels_zeros])
            if self.use_cuda:
                labels = labels.cuda()
            gail_loss = f.binary_cross_entropy_with_logits(pred, labels)
            grad_penalty = self.compute_grad_pen(torch.from_numpy(ones), torch.from_numpy(zeros),
                                                 lambda_=0.05)
            loss = gail_loss + grad_penalty
        else:
            # ####### WGAN loss
            pred_ones = self(ones)
            pred_zeros = self(zeros)
            wgan_loss = torch.mean(pred_zeros) + torch.mean(pred_ones * (-1.))
            graph_penalty = self.compute_graph_pen(torch.from_numpy(zeros_prev), torch.from_numpy(zeros))
            # grad_penalty = self.compute_grad_pen(torch.from_numpy(ones), torch.from_numpy(zeros))
            loss = wgan_loss + graph_penalty

        # loss = torch.mean(- labels * pred.log() - (1 - labels) * (1. - pred).log())
        loss.backward()
        # utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.discriminator_optimizer.step()
        return loss.cpu().detach().numpy()
