import torch
torch.set_num_threads(2)

import bootstrap_policy
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from bootstrap import BDQN


def make_envs(env_name):
    """
    Returns eval and
    :param env_name:
    :return:
    """

    env = make_atari_env(f'{env_name}-v4', n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    eval_env = make_atari_env(f'{env_name}-v4', n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    return env, eval_env

env, eval_env = make_envs("Breakout")

model = BDQN(
    'BCnnPolicy',
    env,
    learning_rate=1.0e-4,
    verbose=1,
    device="cuda:0",
    optimize_memory_usage=True,
    tensorboard_log="./log_benchmark",
    learning_starts=10000, # stub 10k
    buffer_size=int(2e5), # make a bit smaller for testing, memory is a real issue
    use_gamma_function=True
)
model.learn(
    total_timesteps=int(50e6),
    log_interval=50,
    eval_log_path='./eval_benchmark',
    eval_freq=10000,
    eval_env=eval_env,
    tb_log_name='benchmark',
)
