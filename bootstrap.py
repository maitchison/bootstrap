from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn, polyak_update
from bootstrap_policy import BDQNPolicy
from stable_baselines3.common.callbacks import EventCallback

# training should be
# 5m nothing
# 10m 50
# 20m 100
# 50m ~300 (might not get much better than this)

# would be nice to do
#
# GAE instead of 1-step
# masking replay buffer
# mean and median as alternative to mode for evaluation
# for k agents automatically vectorize environment to width of k
# try weight priors (L2 between random initialization and current)
# add output priors (add output of a randomly initialized network to model outputs) as per
#       https://arxiv.org/pdf/1806.03335.pdf is doing right, sounds like RND
#       (this is a great idea, and easy to implement! It'll mean undiscovered states will have high variance.

# Question, relationship between RND and priors... I wonder if there's a paper there? "Random Network Distilation is a
# non-uninform prior.

# todo
# voting during evaluation
# add k value / policy heads
# switch to adam and reference https://arxiv.org/pdf/1806.03335.pdf as justification

# [*] switch agents during rollout
#       (this is done, but first frame is wrong agent, it's hard to fix this though...)
# [*] train all value heads during training
# [ ] get logging going
# [ ] compare DQN vs Bootstrap on Breakout and compare against paper (there are a few changes here... like no prioritized
#        replay

class BDQN(OffPolicyAlgorithm):
    """
    Bootstrap - Deep Q-Network (DQN)

    Paper: https://arxiv.org/pdf/1602.04621.pdf
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param ensemble_k: The number of agents (k) to use in the ensemble.
    :param learning_rate: The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[BDQNPolicy]],
        env: Union[GymEnv, str],
        ensemble_k=10,
        learning_rate: Union[float, Callable] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        use_gamma_function: bool = False,
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            BDQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self.ensemble_k = ensemble_k
        self.current_agent = 0
        self.use_gamma_function = use_gamma_function
        self._last_episode_num = -1

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollout()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

        # switch agents at start of game
        if self._episode_num != self._last_episode_num:
            self.current_agent = (self.current_agent + 1) % self.ensemble_k
            self._last_episode_num = self._episode_num

    def _duplicate_for_ensemble(self, X):
        """
        Takes input and duplicates it ensemble_k times
        :param X: [B, *shape]
        :return: [B*K, *shape]
        """
        B, *shape = X.shape
        K = self.ensemble_k
        X = X[:, np.newaxis] # add extra dim
        if type(X) is np.ndarray:
            X = X.repeat(self.ensemble_k, dim=1)
        elif type(X) is th.Tensor:
            X = X.repeat_interleave(self.ensemble_k, dim=1)
        X = X.reshape([B*K, *shape])
        return X

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        gammas = []
        mean_squareds = []
        variances = []

        B = batch_size
        K = self.ensemble_k
        A = self.action_space.n

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the target Q values [B, K, A]
                q_estimates = self.q_net_target.forward_ensemble(replay_data.next_observations)
                # easier to process as [B*K, A]
                target_q = q_estimates.reshape(B*K, A)
                # Follow greedy policy: use the one with the highest value
                target_q, target_q_indexes = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                extended_rewards = self._duplicate_for_ensemble(replay_data.rewards)
                extended_dones = self._duplicate_for_ensemble(replay_data.dones)

                if self.use_gamma_function:
                    # Calculate gamma based on uncertainty estimate
                    # we use the estimates for the greedy actions selected above

                    ensemble_estimates = q_estimates.reshape([B*K, A])[range(B*K), target_q_indexes] #[B*K]
                    ensemble_estimates = ensemble_estimates.reshape([B, K])
                    variance = th.var(ensemble_estimates, dim=1)  # [B]
                    variances.extend([float(x) for x in variance])
                    mean_squared = th.mean(ensemble_estimates, dim=1)**2
                    mean_squareds.extend([float(x) for x in mean_squared])
                    mean_squared = th.clip(mean_squared, 0.01, float('inf'))
                    rho = -variance / (2*mean_squared)
                    alpha = 1.0 # alpha > 0 --> discount uncertainty
                    gamma = th.exp(alpha * rho)
                    gamma = th.clip(gamma, 0.9, 0.9999) # gamma is [B]
                    gammas.extend([float(x) for x in gamma])
                    gamma = self._duplicate_for_ensemble(gamma) # gamma is [B*K]
                    gamma = gamma.reshape([-1, 1]) # otherwise target_q comes out as [B,B]
                else:
                    gamma = self.gamma

                target_q = extended_rewards + (1 - extended_dones) * gamma * target_q

            # Get current Q estimates
            current_q = self.q_net.forward_ensemble(replay_data.observations)
            current_q = current_q.reshape(-1, self.action_space.n)

            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=self._duplicate_for_ensemble(replay_data.actions.long()))

            # Compute Huber loss (less sensitive to outliers)
            # loss normalization (/= self.ensemble_k) is not required as the 'mean' in huber loss will do this for us
            # this is a side effect of treating this as a larger batch size.
            loss = F.smooth_l1_loss(current_q, target_q)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

        def log_all(X, name):
            logger.record(f"train/{name}_mean", np.mean(X))
            logger.record(f"train/{name}_std", np.std(X))
            logger.record(f"train/{name}_min", np.min(X))
            logger.record(f"train/{name}_max", np.max(X))

        if self.use_gamma_function:
            log_all(gammas, "gamma")
            log_all(mean_squareds, "mean_squared")
            log_all(variances, "variances")

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action = np.array([self.action_space.sample() for _ in range(n_batch)])
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []