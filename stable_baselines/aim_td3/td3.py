import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn, LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.aim_td3.policies import TD3Policy
from stable_baselines.aim_td3.discriminator import Discriminator
from stable_baselines.her.utils import HERGoalEnvWrapper


class AIMTD3(OffPolicyRLModel):
    """
    Twin Delayed DDPG (TD3) with Adversarial Intrinsic Motivation
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=256,
                 disc_train_freq=1000, disc_steps=10, disc_batch_size=256,
                 # disc_train_freq=100, disc_steps=3, disc_batch_size=256,
                 policy_learning_delay=0, reward='aim',
                 tau=0.005, policy_delay=2, action_noise=None,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        super(AIMTD3, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                     policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                     seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        if self.env is not None:
            if not isinstance(self.env, HERGoalEnvWrapper):
                self.env = HERGoalEnvWrapper(self.env)
            self.goal_dim = self.env.goal_dim
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.discriminator_steps = disc_steps
        self.disc_train_freq = disc_train_freq
        self.disc_batch_size = disc_batch_size
        self.rew_std = 1.
        self.rew_mean = 0.
        self.policy_learning_delay = policy_learning_delay
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = 1.
        self.random_exploration_target = random_exploration
        self.random_exploration_schedule = None
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.graph = None
        self.replay_buffer = None
        self.disc_replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None

        self.discriminator = None
        self.irewards = None
        self.irewards_hist = None
        self.max_goal_dist = 0.
        self.reward_type = reward

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.discriminator = Discriminator(x_dim=self.goal_dim * 2, reward_type=self.reward_type)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.disc_replay_buffer = ReplayBuffer(10000)  # self.buffer_size // 300)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                    # Q value when following the current policy
                    qf1_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                            policy_out, reuse=True)

                    tf.summary.histogram('avg_actions', policy_out)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                    # Target policy smoothing, by adding clipped noise to target actions
                    target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                    target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                    # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                    noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                    # Q values when following the target policy
                    qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                noisy_target_action)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.minimum(qf1_target, qf2_target)

                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )

                    # Compute Q-Function loss
                    # qf1_loss = tf.reduce_mean((q_backup - qf1) ** 2)
                    # qf2_loss = tf.reduce_mean((q_backup - qf2) ** 2)
                    qf1_loss = tf.losses.huber_loss(q_backup, qf1, reduction=tf.losses.Reduction.MEAN)
                    qf2_loss = tf.losses.huber_loss(q_backup, qf2, reduction=tf.losses.Reduction.MEAN)

                    qvalues_losses = qf1_loss + qf2_loss

                    # action magnitude loss
                    # action_loss = tf.reduce_mean(tf.maximum(tf.square(policy_out) - 0.8, 0.))
                    action_loss = tf.reduce_mean(tf.square(3. * tf.maximum(tf.abs(policy_out) - 0.8, 0.)))
                    # action_loss = tf.reduce_mean(tf.square(tf.tan(policy_out * np.pi * 0.5)) / 10.)
                    # Policy loss: maximise q value
                    self.policy_loss = policy_loss = -tf.reduce_mean(qf1_pi) + action_loss

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph / 2.)
                    policy_train_op = policy_optimizer.minimize(policy_loss,
                                                                var_list=tf_util.get_trainable_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/values_fn/')

                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Policy param decay
                    policy_params = tf_util.get_trainable_vars('model/pi')
                    self.policy_param_decay = [
                        tf.assign(param, (1. - 1e-4) * param) for param in policy_params
                    ]

                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]


                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss,
                                     qf1, qf2, train_values_op]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()
                self.irewards = tf.placeholder(tf.float64, name='irewards')
                self.irewards_hist = tf.summary.histogram('intr_rewards', self.irewards)

    def _train_step(self, step, writer, learning_rate, update_policy):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
        if np.min(batch_rewards) == -1.:
            batch_dones = batch_rewards + 1.
        else:
            batch_dones = np.copy(batch_rewards)
        disc_obs = batch_next_obs[:, -self.goal_dim * 2:]
        goal_obs = np.copy(disc_obs)
        goal_obs[:, :self.goal_dim] = goal_obs[:, -self.goal_dim:]
        r1 = self.discriminator.reward(disc_obs)
        if self.reward_type == 'aim':
            r1 = (r1 - self.rew_mean) / (self.rew_std * 2.)
        # else:
        #     r1 = (r1 - self.rew_mean)
        drewards = np.squeeze(r1)  # np.squeeze((r1 - self.rew_mean) / (self.rew_std * 2.))

        # disc_obs = batch_obs[:, -self.goal_dim * 2:]
        # r0 = self.discriminator.reward(disc_obs) - self.rew_mean
        # drewards = np.clip(np.squeeze((self.gamma * r1 - r0) / (self.rew_std * 4.)), -.5, .5)

        # self.rew_rms.update(drewards)
        # drewards = (drewards) / (np.sqrt(self.rew_rms.var) * 2.)
        batch_rewards = drewards   # 0.1 * batch_rewards +

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        step_ops = self.step_ops
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.target_ops, self.policy_loss]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
            hist = self.sess.run(self.irewards_hist, feed_dict={self.irewards: drewards})
            writer.add_summary(hist, self.num_timesteps)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qf1_loss, qf2_loss, *_values = out

        return qf1_loss, qf2_loss

    def _train_discriminator(self):
        all_rewards = []
        batch = self.disc_replay_buffer.sample(self.disc_batch_size, env=self._vec_normalize_env)
        _, _, _, target_states, _ = batch
        target_states = target_states[:, -self.goal_dim * 2:]
        target_states[:, :self.goal_dim] = target_states[:, -self.goal_dim:] + np.random.normal(scale=0.01, size=target_states[:, -self.goal_dim:].shape)
        batch = self.disc_replay_buffer.sample(self.disc_batch_size, env=self._vec_normalize_env)
        policy_states, _, _, policy_next_states, _ = batch
        policy_states = policy_states[:, -self.goal_dim * 2:]
        policy_next_states = policy_next_states[:, -self.goal_dim * 2:]
        # goal_dist = np.max(np.sqrt(np.sum(np.square(policy_states[:, :self.goal_dim] -
        #                                             policy_states[:, -self.goal_dim:]),
        #                            axis=-1)))

        # uniformly place the target distribution
        # self.max_goal_dist = max(self.max_goal_dist, goal_dist)
        ###################
        # soften target distribution
        ###################
        # num_rand = self.disc_batch_size // 5
        # random_coordinates = np.random.normal(size=(num_rand, self.goal_dim))
        # random_coordinates = random_coordinates / np.sqrt(np.sum(np.square(random_coordinates), axis=-1, keepdims=True))
        # random_coordinates *= np.random.uniform(low=0, high=goal_dist, size=(num_rand, 1))
        # target_states[-num_rand:, :self.goal_dim] = target_states[-num_rand:, -self.goal_dim:] + random_coordinates
        ###################
        # policy_states[:, :self.goal_dim] += np.random.normal(scale=0.1, size=policy_states[:, :self.goal_dim].shape)
        # Augmenting a few states so that they are near the goal but not the goal
        # num_rand = 25  # self.disc_batch_size // 2
        # policy_states[:num_rand, :self.goal_dim] = policy_states[:num_rand, -self.goal_dim:] + \
        #                                            (np.random.choice([-1., 1.], size=(num_rand, self.goal_dim))) * \
        #                                            (policy_states[:num_rand, :self.goal_dim] -
        #                                             policy_states[:num_rand, -self.goal_dim:])
        # policy_states[:num_rand, :self.goal_dim] = policy_states[:num_rand, -self.goal_dim:] + \
        #                                       (np.random.choice([-1., 1.], size=(num_rand, self.goal_dim))) *\
        #                                       (np.random.rand(num_rand, self.goal_dim) + 0.1) * \
        #                                       (policy_states[:num_rand, :self.goal_dim] -
        #                                        policy_states[:num_rand, -self.goal_dim:])
        loss = self.discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)
        all_rewards.append(self.discriminator.reward(target_states))
        all_rewards.append(self.discriminator.reward(policy_states))
        return loss, all_rewards

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        self.random_exploration_schedule = LinearSchedule(total_timesteps // 30, self.random_exploration_target)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)
            self.disc_replay_buffer = replay_wrapper(self.disc_replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()
            n_updates = 0
            infos_values = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                self.random_exploration = self.random_exploration_schedule.value(step)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    action = self.policy_tf.step(obs[None]).flatten()
                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    unscaled_action = unscale_action(self.action_space, action)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)

                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward

                # Store transition in the replay buffer.
                self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)
                # Store in disc replay buffer
                kwargs = dict(info=info) if self.is_using_her() else {}
                self.disc_replay_buffer.add(obs_, action, reward_, new_obs_, float(done), **kwargs)
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if (self.num_timesteps == self.learning_starts // 2):
                    drewards = []
                    for disc_step in range(self.discriminator_steps):
                        _, rsamples = self._train_discriminator()
                        drewards.extend(rsamples)
                    drewards = np.reshape(drewards, newshape=(-1,))
                    self.rew_std = np.std(drewards) + 0.1
                    self.rew_mean = np.max(drewards) + 0.1

                if (self.num_timesteps > self.learning_starts) and (self.num_timesteps % self.disc_train_freq == 0):
                    drewards = []
                    for disc_step in range(self.discriminator_steps):
                        _, rsamples = self._train_discriminator()
                        drewards.extend(rsamples)
                    drewards = np.reshape(drewards, newshape=(-1,))
                    self.rew_std = np.std(drewards) + 0.1
                    self.rew_mean = np.max(drewards) + 0.1
                    # hist = self.sess.run(self.irewards_hist, feed_dict={self.irewards: drewards})
                    # writer.add_summary(hist, self.num_timesteps)

                # if self.num_timesteps % self.disc_train_freq == 0:
                #     drewards = []
                #     for disc_step in range(self.discriminator_steps):
                #         self._train_discriminator()

                if self.num_timesteps % self.train_freq == 0:
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                                or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        # Note: the policy is updated less frequently than the Q functions
                        # this is controlled by the `policy_delay` parameter
                        pol_update_flag = self.num_timesteps - self.learning_starts >= self.policy_learning_delay
                        pol_update_flag = ((step + grad_step) % self.policy_delay == 0) and pol_update_flag
                        mb_infos_vals.append(
                            self._train_step(step, writer, current_lr, pol_update_flag))

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()

                episode_rewards[-1] += reward_
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                        if writer is not None:
                            summary = tf.Summary(
                                value=[tf.Summary.Value(tag="success_ratio",
                                                        simple_value=episode_successes[-1])])
                            writer.add_summary(summary, self.num_timesteps)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                # substract 1 as we appended a new term just now
                num_episodes = len(episode_rewards) - 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and num_episodes % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
