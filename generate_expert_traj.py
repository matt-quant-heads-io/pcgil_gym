from stable_baselines import DQN
import argparse
from model import FullyConvPolicyBigMap, CustomPolicyBigMap

from utils import make_vec_envs

from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2
import time
import numpy as np

# THIS SECTION IS FOR GEN EXP TRAJ
kwargs_dict = {'resume': False, 'render': False}
log_dir = f'runs/wide'

env_name = f"zelda-wide-v0"
policy = FullyConvPolicyBigMap
env = make_vec_envs(env_name, "wide", log_dir, n_cpu=1, **kwargs_dict)

model = PPO2(policy, env, verbose=1, tensorboard_log=f"./runs/wide")
a_dict = generate_expert_traj(model, 'expert_wide', n_timesteps=int(0), n_episodes=1)
print(a_dict)

numpy_dict = np.load('expert_wide.npz')
print(type(numpy_dict))
print(list(numpy_dict.keys()))

# ['actions', 'obs', 'rewards', 'episode_returns', 'episode_starts']
print(f"ACTIONS")
print(f"=============================")
print(numpy_dict['actions'])
print(numpy_dict['actions'].shape)
print(f"=============================")
print(f"=============================")
print(f"=============================")

print(f"obs")
print(f"=============================")
print(numpy_dict['obs'])
print(numpy_dict['obs'].shape)
print(f"=============================")
print(f"=============================")
print(f"=============================")

print(f"rewards")
print(f"=============================")
print(numpy_dict['rewards'])
print(numpy_dict['rewards'].shape)
print(f"=============================")
print(f"=============================")
print(f"=============================")

print(f"episode_returns")
print(f"=============================")
print(numpy_dict['episode_returns'])
print(numpy_dict['episode_returns'].shape)
print(f"=============================")
print(f"=============================")
print(f"=============================")

print(f"episode_starts")
print(f"=============================")
print(numpy_dict['episode_starts'])
print(numpy_dict['episode_starts'].shape)
print(f"=============================")
print(f"=============================")
print(f"=============================")




