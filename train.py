import torch
torch.set_num_threads(2)

import bootstrap_policy
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from bootstrap import BDQN

import argparse
import uuid


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

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run', type=str, default="test", help="run name")
    parser.add_argument('--device', type=str, default="auto", help="device to use [cpu|cuda:n]")
    parser.add_argument('--beta', type=float, default=0.0, help="random prior beta")
    parser.add_argument('--env', type=str, default="Breakout", help="Atari environment")
    parser.add_argument('--epochs', type=float, default=50.0, help="Number of epochs (million steps)")
    parser.add_argument('--buffer_size', type=int, default=int(2e5), help="Replay buffer size")
    parser.add_argument('--use_gamma', type=bool, default=True)
    parser.add_argument('--k', type=int, default=10)

    return parser.parse_args()


def main():

    args = parse_args()

    env, eval_env = make_envs(args.env)

    id = uuid.uuid4().hex[-8:]
    run_path = f"{args.run} [{id}]"

    model = BDQN(
        'BCnnPolicy',
        env,
        learning_rate=1.0e-4,
        verbose=1,
        device=args.device,
        optimize_memory_usage=True,
        tensorboard_log=f"./{run_path}/log",
        learning_starts=10000,
        buffer_size=args.buffer_size,
        use_gamma_function=args.use_gamma,
        random_prior_beta=args.beta,
        ensemble_k=args.k
    )
    model.learn(
        total_timesteps=int(args.epochs*1e6),
        log_interval=50,
        eval_log_path=f"./{run_path}/eval",
        eval_freq=10000,
        eval_env=eval_env,
        tb_log_name=f"{args.run}",
    )

if __name__ == "__main__":
    main()