"""
Implements a network in Pytorch which outputs the mean and std deviation of a gaussian policy
First I want to implement it with a fixed std dev
"""
import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
import random

# Set random seeds
# seed = 480
# torch.manual_seed(seed)
# random.seed = seed
# np.random.seed = seed


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


class SoftQLearning(nn.Module):
    """
    Learns a soft Q-function. Samples from softmax distribution of Q-values for policy
    """
    def __init__(self, x_dim=1, out_dim=2, max_state=9., min_state=0, ent_coef=0.01, target_update=1e-1):
        super(SoftQLearning, self).__init__()
        self.diff_state = np.array(max_state - min_state).astype(np.float32)
        self.mean_state = np.asarray(self.diff_state / 2 + min_state).astype(np.float32)
        self.input_dim = x_dim
        self.num_actions = out_dim
        self.alpha = ent_coef
        self.q = MlpNetwork(self.input_dim, output_dim=out_dim, n_units=64)
        self.q_target = MlpNetwork(self.input_dim, output_dim=out_dim, n_units=64)
        self.target_params = self.q_target.parameters()
        self.q_params = self.q.parameters()
        self.target_update_rate = target_update

    def parameters(self, recurse: bool = True):
        return self.q_params

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        normalize input
        :param x:
        :return:
        """
        x = x.type(torch.float32)
        x = (x - self.mean_state) / self.diff_state
        return x

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        run policy and value functions and return
        :param x:
        :return:
        """
        x = self.normalize(x)
        q = self.q(x)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1)
        # self.alpha = max(0.01, 0.99 * self.alpha + 0.01 * (torch.mean(torch.abs(q)).detach().numpy() / 10.))
        qt = self.q_target(x)
        vt = self.alpha * torch.logsumexp(qt / self.alpha, dim=-1)
        return q, v, qt, vt

    # def pi_loss(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    #     """
    #     Return log_pi for the policy gradient
    #     :param x:
    #     :param actions:
    #     :return:
    #     """
    #     x = self.normalize(x)
    #     logits = self.pi(x)
    #     actions = actions.type(torch.long)
    #     log_pi = logits.gather(dim=-1, index=actions)
    #     return log_pi

    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from policy
        :param x:
        :return:
        """
        x = self.normalize(x)
        q = self.q(x)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1)
        logits = 1. / self.alpha * (q - v)
        pi = torch.exp(logits)
        action = pi.multinomial(1)
        return action

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return entropy
        """
        x = self.normalize(x)
        q = self.q(x)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1)
        logits = 1. / self.alpha * (q - torch.unsqueeze(v, dim=-1))
        entropy_kl = torch.sum(torch.log(torch.ones_like(logits)/self.num_actions) - logits, dim=-1)
        # pi = torch.exp(logits)
        # pisum = torch.sum(pi, dim=-1)
        # entropy = -torch.sum(pi * logits, dim=-1)
        return entropy_kl

    def update_target(self):
        with torch.no_grad():
            for c, t in zip(self.q.parameters(), self.q_target.parameters()):
                t.data.copy_((1. - self.target_update_rate) * t.data + self.target_update_rate * c.data)


class NormalPolicy(nn.Module):
    """
    Policy for 1-d continuous actions
    """
    def __init__(self, x_dim=1):
        super(NormalPolicy, self).__init__()
        self.input_dim = x_dim
        self.mu = MlpNetwork(self.input_dim, output_nonlinearity=torch.tanh)
        self.v = MlpNetwork(self.input_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        run policy and value functions and return
        :param x:
        :return:
        """
        mu = self.mu(x)
        v = self.v(x)
        return mu, v

    def pi_loss(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log_pi for the policy gradient
        :param x:
        :param actions:
        :return:
        """
        mu = self.mu(x)
        log_pi = (actions - mu) ** 2
        return log_pi

    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from policy
        :param x:
        :return:
        """
        mu = self.mu(x)
        action = torch.normal(mean=mu, std=torch.ones_like(mu))
        return action


class NoCriticPolicy(nn.Module):
    """
    Policy for 1-d continuous actions
    """
    def __init__(self, x_dim=1, out_dim=2, max_state=9., min_state=0, ent_coef=0.01):
        super(NoCriticPolicy, self).__init__()
        self.mean_state = np.asarray((max_state - min_state) / 2 + min_state).astype(np.float32)
        self.diff_state = np.array(max_state - min_state).astype(np.float32)
        self.input_dim = x_dim
        self.num_actions = out_dim
        self.alpha = ent_coef
        self.logits = MlpNetwork(self.input_dim, output_dim=out_dim, n_units=32, output_nonlinearity=f.log_softmax)
        self.p_params = self.logits.parameters()

    def parameters(self, recurse: bool = True):
        return self.p_params

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        normalize input
        :param x:
        :return:
        """
        x = x.type(torch.float32)
        x = (x - self.mean_state) / self.diff_state
        return x

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        run policy and value functions and return
        :param x:
        :return:
        """
        l = self.logits(x)
        return l

    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from policy
        :param x:
        :return:
        """
        pi = torch.exp(self(x))
        action = pi.multinomial(1)
        return action

    def pi_loss(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log_pi for the policy gradient
        :param x:
        :param actions:
        :param advantage:
        :return:
        """
        l = self(x)
        actions = actions.type(torch.long)
        loglikelihood = l.gather(dim=-1, index=actions)
        return loglikelihood

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return entropy
        """
        logits = self(x)
        entropy_kl = torch.sum(torch.log(torch.ones_like(logits)/self.num_actions) - logits, dim=-1)
        # pi = torch.exp(logits)
        # pisum = torch.sum(pi, dim=-1)
        # entropy = -torch.sum(pi * logits, dim=-1)
        return entropy_kl


class SoftmaxPolicy(nn.Module):
    """
    Policy for discrete actions
    """
    def __init__(self, x_dim=1, out_dim=2, max_state=9., min_state=0):
        super(SoftmaxPolicy, self).__init__()
        self.mean_state = np.asarray((max_state - min_state) / 2 + min_state).astype(np.float32)
        self.diff_state = np.array(max_state - min_state).astype(np.float32)
        self.input_dim = x_dim
        self.num_actions = out_dim
        self.pi = MlpNetwork(self.input_dim, output_dim=out_dim, output_nonlinearity=f.log_softmax)
        self.v = MlpNetwork(self.input_dim)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        normalize input
        :param x:
        :return:
        """
        x = x.type(torch.float32)
        x = (x - self.mean_state) / self.diff_state
        return x

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        run policy and value functions and return
        :param x:
        :return:
        """
        x = self.normalize(x)
        logits = self.pi(x)
        v = self.v(x)
        return logits, v

    def pi_loss(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log_pi for the policy gradient
        :param x:
        :param actions:
        :return:
        """
        x = self.normalize(x)
        logits = self.pi(x)
        actions = actions.type(torch.long)
        log_pi = logits.gather(dim=-1, index=actions)
        return log_pi

    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from policy
        :param x:
        :return:
        """
        x = self.normalize(x)
        logits = self.pi(x)
        pi = torch.exp(logits)
        action = pi.multinomial(1)
        return action

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return entropy
        """
        x = self.normalize(x)
        logits = self.pi(x)
        entropy_kl = torch.sum(torch.log(torch.ones_like(logits)/self.num_actions) - logits, dim=-1)
        # pi = torch.exp(logits)
        # entropy = -torch.sum(pi * logits, dim=-1)
        return entropy_kl
