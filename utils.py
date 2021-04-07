import os
# from . import TILES_MAP # Add pcgil to path
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl import wrappers
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

import numpy as np


CHAR_MAP = {"door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'}

# TODO: This is a placeholder mapping (confirm correct mapping with Ahmed)
ONEHOT_MAP = {"door": [1, 0, 0, 0, 0, 0, 0, 0],
              "key": [0, 1, 0, 0, 0, 0, 0, 0],
              "player": [0, 0, 1, 0, 0, 0, 0, 0],
              "bat": [0, 0, 0, 1, 0, 0, 0, 0],
              "spider": [0, 0, 0, 0, 1, 0, 0, 0],
              "scorpion": [0, 0, 0, 0, 0, 1, 0, 0],
              "solid": [0, 0, 0, 0, 0, 0, 1, 0],
              "empty": [0, 0, 0, 0, 0, 0, 0, 1]}

TILES_MAP = {"g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"}

# TODO: This is a placeholder mapping (confirm correct mapping with Ahmed)
INT_MAP = {
    "empty": 0,
    "solid": 1,
    "door": 2,
    "key": 3,
    "player": 4,
    "bat": 5,
    "spider": 6,
    "scorpion": 7
}


def hamming_distance_pct(map1, map2):
    map1_str = ''
    map2_str = ''
    for row_i in range(len(map1)):
        for col_i in range(len(map1[0])):
            map1_str += CHAR_MAP[map1[row_i][col_i]]
            map2_str += CHAR_MAP[map2[row_i][col_i]]
            print()
    num_char_diffs = 0
    for idx, char in enumerate(map1_str):
        if char == map2_str[idx]:
            continue
        else:
            num_char_diffs += 1

    return num_char_diffs / len(map1_str)


def convert_action_to_npz_format(x , y, action):
    idx = (x * 11 * 8 + y * 8) + INT_MAP[action]
    return idx


def str_map_to_onehot(str_map):
    new_map = str_map.copy()
    for row_i in range(len(str_map)):
        for col_i in range(len(str_map[0])):
            new_tile = [0]*8
            new_tile[INT_MAP[str_map[row_i][col_i]]] = 1
            new_map[row_i][col_i] = new_tile
    return new_map


def int_map_to_onehot(int_map):
    new_map = int_map.copy()
    for row_i in range(len(int_map)):
        for col_i in range(len(int_map[0])):
            new_tile = [0]*8
            new_tile[int_map[row_i][col_i]] = 1
            new_map[row_i][col_i] = np.array(new_tile)
    return np.array(new_map)


class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)


def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk


def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env


def convert_ob_to_int_arr(ob_dict):
    import numpy as np
    ob = ob_dict['map']
    new_map = []
    for row_i in range(len(ob)):
        new_row = []
        for col_i in range(len(ob[0])):
            new_row.append(INT_MAP[ob[row_i][col_i]])
        new_map.append(new_row)
    return new_map


def to_2d_array_level(file_name):
    level = []

    with open(f'{os.path.dirname(__file__)}/good_levels_set/{file_name}', 'r') as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != '\n':
                    new_row.append(TILES_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1: len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1: len(row) - 1]
        level.append(new_row)
    return level