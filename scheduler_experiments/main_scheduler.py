"""
GAIL file
"""
import numpy as np
import torch
from torch import nn

# from torch.nn import utils
import torch.nn.functional as f
import random
from policy import MlpNetwork, SoftQLearning
from scheduler_mdp import SchedulerMDP
from rnd import RND
import matplotlib.pyplot as plt
from buffers import ReplayBuffer
import argparse
import os
import wandb
from os import path
import pandas as pd


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", help="random seed", type=int, default=1123)
parser.add_argument("--num_epochs", help="random seed", type=int, default=200)
parser.add_argument(
    "--rnd", help="random network distillation", type=bool, default=False
)
parser.add_argument(
    "--reward",
    help="reward function to use ['gail', 'airl', 'fairl', 'aim', 'none']",
    type=str,
    default="aim",
)
parser.add_argument(
    "--dir", help="directory to save results in", type=str, default="aim_results"
)
args = parser.parse_args()
torch.set_default_dtype(torch.float32)
# Set random seeds
seed = 42 * args.seed
print(args.seed)
torch.manual_seed(seed)
random.seed = seed
np.random.seed = seed
reward_to_use = args.reward  # use one of ['gail', 'airl', 'fairl', 'none']
print(reward_to_use)


def wasserstein_reward(d):
    """
    return the wasserstein reward
    """
    return d


def gail_reward(d):
    """
    Take discriminaotr output and return the gail reward
    :param d:
    :return:
    """
    d = torch.sigmoid(d)
    return d.log()  # - (1 - d).log()


def airl_reward(d):
    """
    Take discriminaotr output and return AIRL reward
    :param d:
    :return:
    """
    s = torch.sigmoid(d)
    reward = s.log() - (1 - s).log()
    return reward


def fairl_reward(d):
    """
    Take discriminator output and return FAIRL reward
    :param d:
    :return:
    """
    d = torch.sigmoid(d)
    h = d.log() - (1 - d).log()
    return h.exp() * (-h)


reward_dict = {
    "gail": gail_reward,
    "airl": airl_reward,
    "fairl": fairl_reward,
    "aim": wasserstein_reward,
    "none": None,
}


class Discriminator(nn.Module):
    """
    The discriminator used to learn the potentials or the reward functions
    """

    def __init__(self, x_dim=1, max_state=10.0, min_state=0):
        super(Discriminator, self).__init__()
        self.mean_state = torch.tensor(
            (max_state - min_state) / 2 + min_state, dtype=torch.float32
        )
        self.diff_state = torch.tensor(max_state - min_state, dtype=torch.float32)
        self.input_dim = x_dim
        self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)

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
        return discriminator output
        :param x:
        :return:
        """
        x = self.normalize(x)
        output = self.d(x)
        return output


def to_one_hot(x: torch.Tensor, num_vals) -> torch.Tensor:
    """
    Convert tensor to one-hot encoding
    """
    if type(x) is not torch.Tensor:
        x = torch.tensor(x)
    x = x.type(torch.long)
    x_one_hot = torch.zeros((x.shape[0], num_vals), dtype=torch.float32)
    x_one_hot = x_one_hot.scatter(1, x, 1.0)
    return x_one_hot


class GAIL:
    """
    Class to take the continuous MDP and use gail to match given target distribution
    """

    def __init__(self):
        self.env = SchedulerMDP()
        self.policy = SoftQLearning(
            x_dim=self.env.dims,
            out_dim=len(self.env.action_space),
            max_state=self.env.max_state,
            min_state=self.env.min_state,
            ent_coef=0.3,
            target_update=3e-2,
        )
        self.discriminator = Discriminator(
            x_dim=self.env.dims,
            max_state=self.env.max_state,
            min_state=self.env.min_state,
        )
        self.discount = 0.99
        self.check_state = set()
        self.agent_buffer = ReplayBuffer(size=5000)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())  # , lr=3e-4)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters()
        )  # , lr=1e-4)
        if args.rnd:
            self.rnd = RND(x_dim=self.env.dims)
        else:
            self.rnd = None
        self.max_r = 1.0
        self.min_r = -1.0

    def gather_data(self, num_trans=100) -> None:
        """
        Gather data from current policy
        used to:
        * fit value function
        * update policy
        * plot histograms
        :param num_trans:
        :return:
        """
        t = 0
        while t < num_trans:
            s = self.env.reset()
            s = torch.tensor(s).type(torch.float32).reshape([-1, self.env.dims])
            done = False
            while not done:
                # self.states.append(deepcopy(s))
                action = self.policy.sample_action(s)
                # self.actions.append(a)
                a = np.squeeze(action.data.detach().numpy())
                s_p, r, done, _ = self.env.step(a)
                s_p = torch.tensor(s_p).type(torch.float32).reshape([-1, self.env.dims])
                # d = self.discriminator(sp)
                # i_r = gail_reward(d)
                # self.next_states.append(deepcopy(s))
                # self.rewards.append(i_r)  # deepcopy(r))
                # self.dones.append(deepcopy(done))
                self.agent_buffer.add(
                    s.squeeze(), action.reshape([-1]).detach(), r, s_p.squeeze(), done
                )
                # if s_p not in self.check_state:
                #     self.check_state.add(s_p)
                #     self.target_buffer.add(s, a, r, s_p, done)
                s = s_p
                t += 1
            # self.states.append(s)

    def compute_td_targets(self, states, next_states, dones, rewards=None):
        """
        Compute the value of the current states and
        the TD target based on one step reward
        and value of next states
        :return: value of current states v, TD target targets
        """
        states = states.reshape([-1, self.env.dims])
        next_states = next_states.reshape([-1, self.env.dims])
        v = self.policy(states)[0]
        v_prime = self.policy(next_states)[-1]
        if rewards is not None:
            dones = rewards.type(torch.float32).reshape([-1, 1])
        else:
            dones = dones.type(torch.float32).reshape([-1, 1])
        reward_func = reward_dict[reward_to_use]
        if reward_func is not None:
            # d0 = self.discriminator(states)
            d1 = self.discriminator(next_states)
            # Compute rewards
            # r0 = reward_func(d0)
            r1 = reward_func(d1)
            rewards = rewards.type(torch.float32).reshape([-1, 1]) + (
                (r1 - self.max_r) / (self.max_r - self.min_r)
            )
        targets = rewards.type(torch.float32).reshape([-1, 1])
        targets += (1.0 - dones) * self.discount * v_prime.reshape([-1, 1])
        return v, targets.detach()

    def fit_v_func(self):
        """
        This function will train the value function using the collected data
        :return:
        """
        self.policy_optimizer.zero_grad()
        s, a, r, s_p, dones = self.agent_buffer.sample(100)
        if args.rnd:
            spn = s_p.detach().numpy()
            self.rnd.update(spn)
            r += 0.3 * self.rnd.reward(spn)

        q, targets = self.compute_td_targets(s, s_p, dones, rewards=r)
        actions = torch.tensor(a, dtype=torch.long)
        v = q.gather(dim=-1, index=actions)
        loss = torch.mean(0.5 * (targets - v) ** 2)
        loss.backward()
        self.policy_optimizer.step()
        self.policy.update_target()
        return

    # def optimize_policy(self):
    #     """
    #     This function will optimize the policy to maximize returns
    #     Based on collected data
    #     :return:
    #     """
    #     self.policy_optimizer.zero_grad()
    #     s, a, r, s_p, dones = self.agent_buffer.sample(100)
    #     v, targets = self.compute_td_targets(s, s_p, dones, rewards=r)
    #     advantages = (targets - v).detach()
    #     a = a.reshape([-1, 1]).detach()
    #     neg_log_pi = -1. * self.policy.pi_loss(s.reshape([-1, self.env.dims]), a)
    #     entropy_kl = self.policy.entropy(s.reshape([-1, self.env.dims]))
    #     loss = torch.mean(advantages * neg_log_pi) + 1e-1 * torch.mean(entropy_kl)
    #     loss.backward()
    #     self.policy_optimizer.step()
    #     return

    def compute_aim_pen(
        self,
        target_state: torch.Tensor,
        prev_state: torch.Tensor,
        next_state_state: torch.Tensor,
        lambda_=10.0,
    ):
        """
        Computes values of the discriminator at different points
        and constraints the difference to be 0.1
        """
        prev_out = self.discriminator(prev_state)
        next_out = self.discriminator(next_state_state)
        penalty = (
            lambda_
            * torch.max(torch.abs(next_out - prev_out) - 0.1, torch.tensor(0.0))
            .pow(2)
            .mean()
        )
        return penalty

    def compute_grad_pen(
        self, target_state: torch.Tensor, policy_state: torch.Tensor, lambda_=10.0
    ):
        """
        Computes the gradients by mixing the data randomly
        and creates a loss for the magnitude of the gradients.
        """
        alpha = torch.rand(target_state.size(0), 1)
        # expert_data = torch.cat([expert_state, expert_action], dim=1)
        # policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(target_state).to(target_state.device)

        mixup_data = alpha * target_state + (1 - alpha) * policy_state
        mixup_data.requires_grad = True

        disc = self.discriminator(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (
            lambda_
            * (torch.max(grad.norm(2, dim=1) - 0.01, torch.tensor(0.0))).pow(2).mean()
        )
        return grad_pen

    def optimize_discriminator(self):
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        num_samples = 100
        self.discriminator_optimizer.zero_grad()

        # gather target distribution states
        # _, _, _, target_distribution, _ = self.target_buffer.sample(100)
        target_distribution = np.array(
            [self.env.sample_target_state() for _ in range(num_samples)]
        )
        states, _, _, next_states, _ = self.agent_buffer.sample(num_samples)
        target_distribution = target_distribution.reshape([-1, 1])
        next_states = next_states.reshape([-1, self.env.dims])
        ones = (
            torch.tensor(target_distribution)
            .type(torch.float32)
            .reshape([-1, self.env.dims])
        )
        zeros = (
            torch.tensor(next_states).type(torch.float32).reshape([-1, self.env.dims])
        )
        zeros_prev = (
            torch.tensor(states).type(torch.float32).reshape([-1, self.env.dims])
        )

        # ########## GAIL loss
        if reward_to_use != "aim":
            labels_ones = torch.ones((num_samples, 1)) * 0.9
            labels_zeros = torch.ones((num_samples, 1)) * 0.1
            data = torch.cat([ones, zeros])
            pred = self.discriminator(data)
            labels = torch.cat([labels_ones, labels_zeros])
            gail_loss = f.binary_cross_entropy_with_logits(pred, labels)
            grad_penalty = self.compute_grad_pen(ones, zeros)
            loss = gail_loss + grad_penalty
        else:
            # ####### WGAN loss
            pred_ones = self.discriminator(ones)
            pred_zeros = self.discriminator(zeros)
            preds = torch.cat([pred_zeros, pred_ones], dim=0)
            self.max_r = torch.max(preds).detach().cpu().numpy() + 0.1
            self.min_r = torch.min(preds).detach().cpu().numpy() - 0.1
            wgan_loss = torch.mean(pred_zeros) + torch.mean(pred_ones * (-1.0))
            aim_penalty = self.compute_aim_pen(ones, zeros_prev, zeros)
            # grad_penalty = self.compute_grad_pen(ones, zeros)
            loss = wgan_loss + aim_penalty  # + grad_penalty

        # loss = torch.mean(- labels * pred.log() - (1 - labels) * (1. - pred).log())
        loss.backward()
        # utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.discriminator_optimizer.step()

    # def plot_dist(self, num_samples=100, it=0, dname='aim'):
    #     """
    #     plot the two distributions as histograms
    #     :return:
    #     """
    #     # dname = 'r_neg'
    #     if not path.exists(dname):
    #         os.mkdir(dname)

    #     # _, _, _, target_distribution, _ = self.target_buffer.sample(num_samples)
    #     states, _, _, next_states, _ = self.agent_buffer.sample(num_samples)
    #     target_dist = np.reshape(self.env.target_distribution(), (-1,))
    #     target_distribution = np.random.choice(target_dist.shape[0], num_samples, p=target_dist)
    #     target_distribution = target_distribution.reshape([-1, 1]).astype(np.float32)
    #     if self.env.dims > 1:
    #         target_distribution = np.concatenate([target_distribution, target_distribution], axis=-1)
    #         target_distribution[:, 0] = target_distribution[:, 0] // self.env.y_dim
    #         target_distribution[:, 1] = target_distribution[:, 1] % self.env.y_dim
    #     # target_distribution += np.random.normal(loc=0, scale=0.5, size=target_distribution.shape)
    #     next_states = next_states.numpy().reshape([-1, self.env.dims]).astype(np.float32)
    #     # next_states += np.random.normal(loc=0., scale=0.01, size=next_states.shape)
    #     q, v, qt, vt = self.policy(states)
    #     print(f"q: {np.mean(q.detach().numpy())}, v: {np.mean(v.detach().numpy())},"
    #           f" qt: {np.mean(qt.detach().numpy())}, vt: {np.mean(vt.detach().numpy())}")
    #     if self.env.dims == 1:
    #         xloc = np.arange(0, self.env.num_states)
    #         target_distribution = to_one_hot(target_distribution, self.env.num_states).numpy()
    #         plt.bar(xloc, np.sum(target_distribution, axis=0), color='r', alpha=0.3, label='target')
    #         next_states = to_one_hot(next_states, self.env.num_states).numpy()
    #         plt.bar(xloc, np.sum(next_states, axis=0), color='b', alpha=0.3, label='agent')
    #         for t in self.env.target_state:
    #             plt.axvline(x=t, color='r', linestyle='dashed', linewidth=2)
    #         # sns.kdeplot(np.squeeze(target_distribution), shade=True, color='r', shade_lowest=False, alpha=0.3,
    #         #             label='target')
    #         # sns.kdeplot(np.squeeze(next_states), shade=True, color='b', shade_lowest=False, alpha=0.3,
    #         #             label='agent')
    #     else:
    #         from matplotlib.ticker import AutoMinorLocator
    #         target_vals, target_counts = np.unique(target_distribution, axis=0, return_counts=True)
    #         agent_vals, agent_counts = np.unique(next_states, axis=0, return_counts=True)
    #         target_counts = target_counts.astype(np.float) / np.max(target_counts)
    #         agent_counts = agent_counts.astype(np.float) / np.max(agent_counts)
    #         # for it in range(target_counts.shape[0]):
    #         #     plt.plot(target_vals[it, 0] + 0.5, target_vals[it, 1] + 0.5, marker_size=40 * target_counts[it],
    #         #              color='r', alpha=0.2)
    #         # for ia in range(agent_counts.shape[0]):
    #         #     plt.plot(agent_vals[ia, 0] + 0.5, agent_vals[ia, 1] + 0.5, marker_size=40 * agent_counts[ia],
    #         #              color='b', alpha=0.2)

    #         plt.xlim(left=0., right=self.env.x_dim)
    #         plt.ylim(bottom=0., top=self.env.y_dim)
    #         plt.scatter(target_vals[:, 0] + 0.5, target_vals[:, 1] + 0.5, 200 * target_counts,
    #                     color='r', alpha=0.5, label='target')
    #         plt.scatter(agent_vals[:, 0] + 0.5, agent_vals[:, 1] + 0.5, 200 * agent_counts,
    #                     color='b', alpha=0.5, label='agent')
    #         plt.xticks(np.arange(self.env.x_dim) + 0.5, np.arange(self.env.x_dim))
    #         plt.yticks(np.arange(self.env.y_dim) + 0.5, np.arange(self.env.y_dim))
    #         minor_locator = AutoMinorLocator(2)
    #         plt.gca().xaxis.set_minor_locator(minor_locator)
    #         plt.gca().yaxis.set_minor_locator(minor_locator)
    #         plt.gca().set_aspect('equal')
    #         plt.grid(which='minor')
    #         # sns.kdeplot(target_distribution[:, 0], target_distribution[:, 1],
    #         # shade=True, color='r', shade_lowest=False,
    #         #             alpha=0.5, label='target')
    #         # sns.kdeplot(next_states[:, 0], next_states[:, 1], shade=True, color='b', shade_lowest=False, alpha=0.5,
    #         #             label='agent')
    #     plt.legend()
    #     # plt.hist(target_distribution, bins=10, alpha=0.4, color='red')
    #     # plt.hist(next_states, bins=10, alpha=0.4, color='blue')
    #     # plt.axvline(x=self.env.target_state, color='r', linestyle='dashed', linewidth=2)
    #     # plt.legend(['target', 'agent'])
    #     plt.title(f'Density for agent and target distributions state Iteration {it}')
    #     # plt.show()
    #     plt.tight_layout()
    #     plt.savefig(f'{dname}/d_{it // 10}.png', dpi=300)
    #     # exit()
    #     plt.cla()
    #     plt.clf()

    #     reward_func = reward_dict[reward_to_use]
    #     if reward_func is not None:
    #         r_states = []
    #         for ia in range(self.env.x_dim):
    #             for ja in range(self.env.y_dim):
    #                 r_states.append([ia, ja])
    #         r_states = np.asarray(r_states)
    #         r_states = torch.tensor(r_states)
    #         d = reward_func(self.discriminator(r_states)).detach().numpy()
    #         print(f'Max potential: {np.max(d)}, Min potential: {np.min(d)}')
    #         # Compute rewards
    #         rewards = (d - self.max_r) / (self.max_r - self.min_r)
    #         rewards = np.reshape(rewards, newshape=(self.env.x_dim, self.env.y_dim))
    #         # Flip matrix so the rewards image is plotted aligned with the distribution
    #         # grid above
    #         # rewards = np.flip(rewards, axis=0)
    #         rewards = np.transpose(rewards)
    #         plt.imshow(rewards, cmap='magma', origin='lower')
    #         plt.colorbar()
    #         plt.title(f'Rewards at Iteration {it}')
    #         plt.tight_layout()
    #         plt.savefig(f'{dname}/r_{it // 10}.png', dpi=300)
    #         plt.cla()
    #         plt.clf()
    #     entropy_kl = self.policy.entropy(states.reshape([-1, self.env.dims]))
    #     entropy_kl = np.mean(entropy_kl.detach().numpy())
    #     print(f"Entropy KL at Iteration {it} is {entropy_kl}")

    #     if self.env.d == 1:
    #         states = np.arange(0, self.env.num_states)
    #         s = torch.tensor(states).type(torch.float32).reshape([-1, 1])
    #         # s = to_one_hot(s, self.env.num_states)
    #         d = self.discriminator(s)
    #         reward_func = reward_dict[reward_to_use]
    #         if reward_func is not None:
    #             rewards = reward_func(d).squeeze().detach().numpy()
    #             plt.cla()
    #             plt.bar(states, rewards, width=0.5)
    #             plt.xlabel('states')
    #             plt.ylabel('rewards')
    #             plt.title('Rewards for entering state')
    #             plt.show()
    #             logits = self.policy(s)[0]
    #             policy = torch.exp(logits).detach().numpy()
    #             plt.bar(states, policy[:, 0], width=0.5)
    #             plt.xlabel('states')
    #             plt.ylabel('P(left|state)')
    #             plt.title('Policy')
    #             plt.show()
    #         plt.cla()


if __name__ == "__main__":
    wandb.init(project="aim", config=vars(args), name=args.reward)

    gail = GAIL()
    gail.gather_data(num_trans=500)

    for i in range(args.num_epochs):
        if reward_to_use != "none":
            for _ in range(5):
                # gail.gather_data(num_trans=500)
                gail.optimize_discriminator()

        for _ in range(10):
            gail.gather_data(num_trans=500)
            gail.fit_v_func()

        # extract rewards from the gathered data
        returns = gail.env._returns_buffer
        wandb.log({"returns": returns[-1], "iteration": i + 1})
        if i == args.num_epochs - 1:
            logdir = f"results/{args.reward}/{args.seed}.csv"
            if not path.exists(f"results/{args.reward}"):
                os.makedirs(f"results/{args.reward}")
            df = pd.DataFrame({"returns": returns, "t": np.arange(len(returns))})
            df.to_csv(logdir, index=False)
