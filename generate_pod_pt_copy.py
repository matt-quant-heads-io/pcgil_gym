import sys
import argparse

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image

import numpy as np
import random

from utils import to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format, string_from_2d_arr, int_arr_from_str_arr

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--level", help="Level from which to generate pod", type=int)
parser.add_argument("-n", "--num_pods", help="Number of pods to generate", type=int)
parser.add_argument("-r", "--render", help="Determines whether to render pod iterations", type=str)
parser.add_argument("--rep", help="The representation to use", type=str)


def render_map(map, prob, rep, save=False, filename=''):
    # format image of map for rendering
    img = prob.render(map)
    img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    img = np.array(img)
    if save:
        im = Image.fromarray(img)
        im.save(filename)
    return img


def generate_play_trace_wide(map, prob, rep, actions_list, render=False):
    play_trace = []
    # loop through from 0 to 19 (for 20 tile change actions)
    old_map = map.copy()
    for num_tc_action in range(14):
        new_map = old_map.copy()
        transition_info_at_step = [None, old_map, None, None]
        actions = actions_list.copy()
        row_idx, col_idx = random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1)
        new_map[row_idx] = old_map[row_idx].copy()
        transition_info_at_step[2] = (row_idx, col_idx)
        # 1) get the tile type at row_i, col_i and remove it from the list
        old_tile_type = map[row_idx][col_idx]
        transition_info_at_step[3] = old_tile_type
        # 2) remove the tile from the actions_list
        actions.remove(old_tile_type)
        # 3) select an action from the list
        new_tile_type = random.choice(actions)
        # 4) update the map with the new tile type
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map
        play_trace.append(transition_info_at_step)

        old_map = new_map

        # Render
        if render:
            map_img = render_map(new_map, prob, rep)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            input(f'')
            ren.close()
    return play_trace


def generate_play_trace_narrow(map, prob, rep, actions_list, render=False):
    play_trace = []
    old_map = map.copy()
    tile_coors = []
    # creates an array of tile coordinates, shuffles them to the order in which they are to be
    # modified, can change to work from left to right if necessary, following tree search paper
    # where these concepts were first introduced
    for row in map:
        for col in map:
            tile_coors.append((row, col))
    tile_coors = np.array(tile_coors)
    tile_coors = np.flip(tile_coors)
    # random.shuffle(tile_coors)

    # change from left-right, to bottom right-top left
    for num_tc_action in range(len(tile_coors)):
        new_map = old_map.copy()
        transition_info_at_step = [None, old_map, None, None]
        actions = actions_list.copy()
        row_idx, col_idx = tile_coors[num_tc_action][0], tile_coors[num_tc_action][1]
        new_map[row_idx] = old_map[row_idx].copy()
        transition_info_at_step[2] = (row_idx, col_idx)
        # 1) get the tile type at row_i, col_i and remove it from the list
        old_tile_type = map[row_idx][col_idx]
        # 2) remove the tile from the actions_list
        # actions.remove(old_tile_type) commented out because some will not change
        # 3) select an action from the list
        new_tile_type = random.choice(actions)
        # 4) update the map with the new tile type
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map
        play_trace.append(transition_info_at_step)

        old_map = new_map

        # Render
        if render:
            map_img = render_map(new_map, prob, rep)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            input(f'')
            ren.close()
    return play_trace


def generate_play_trace_turtle(map, prob, rep, actions_list, render=False):
    pass


def get_numpy_dict_from_play_trace_wide(play_trace, pcgrl_env):
    actions = []
    obs = []
    rewards = []
    episode_returns = []
    episode_starts = [True]
    # episode_starts = [np.array([True])]
    episode_return = 0.0
    pcgrl_env.reset()
    pcgrl_env._rep.set_map(np.array(int_arr_from_str_arr(play_trace[0][1])))
    actions.append([convert_action_to_npz_format(play_trace[0][2][0], play_trace[0][2][1], play_trace[0][-1],
                                          int_map_to_onehot(pcgrl_env._rep.get_observation()['map']))])
    rewards.append([0.0])
    obs.append(int_map_to_onehot(pcgrl_env._rep.get_observation()['map']))
    for tuple_idx, pt_tuple in enumerate(play_trace):
        episode_starts.append(np.array([False]))
        # episode_starts.append(np.array([False]))
        observation, reward, done, info = pcgrl_env.step([pt_tuple[2][1], pt_tuple[2][0], pcgrl_env._prob.get_tile_types().index(pt_tuple[-1])])
        ob = observation['map']
        ob_oh = int_map_to_onehot(ob)
        obs.append(ob_oh)
        action = convert_action_to_npz_format(pt_tuple[2][0], pt_tuple[2][1], pt_tuple[-1], int_map_to_onehot(observation['map']))
        actions.append([action])
        reward = pcgrl_env._prob.get_reward(pcgrl_env._prob.get_stats(pt_tuple[0]), pcgrl_env._prob.get_stats(pt_tuple[1]))
        rewards.append([reward])
        episode_return += reward
    episode_returns.append(episode_return)
    actions = np.array(actions)
    obs = np.array(obs)
    rewards = np.array(rewards)
    episode_returns = np.array(episode_returns)
    episode_starts = np.array(episode_starts)

    # print(f"actions: {len(actions)}")
    # print(f"len of obs: {len(obs)}")
    # print(f"len of rewards: {len(rewards)}")
    # print(f"len of episode_returns: {len(episode_returns)}")
    # print(f"len of episode_starts: {len(episode_starts)}")

    # print({
    #     'actions': actions,
    #     'obs': obs,
    #     'rewards': rewards,
    #     'episode_returns': episode_returns,
    #     'episode_starts': episode_starts
    # })

    return {
        'actions': actions,
        'obs': obs,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }


def main():
    args = parser.parse_args()
    level = args.level
    render = args.render
    render = True if render == 't' else False
    rep = args.rep
    rep_as_str = rep
    num_pods = args.num_pods

    prob = ZeldaProblem()
    rep = REPRESENTATIONS[rep]()
    actions_list = [act for act in list(TILES_MAP.values())]

    if level == -1:
        FILENAME_TEMPLATE = "zelda_lvl{}.txt"
        if rep_as_str == 'wide':
            complete_numpy_dict = {'actions': [], 'obs': [], 'rewards': [],
                                   'episode_returns': [], 'episode_starts': []}
            for idx in range(50):
                filename = FILENAME_TEMPLATE.format(idx)
                for idx_j in range(num_pods):
                    map = to_2d_array_level(filename)
                    play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=render)
                    pcgrl_env = PcgrlEnv("zelda", "wide")
                    # numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                    numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, pcgrl_env)
                    for k, v in numpy_dict.items():
                        complete_numpy_dict[k].extend(numpy_dict[k])
            np.savez(f'expert_trajectories/wide/expert_zelda_complete', **complete_numpy_dict)
        elif rep_as_str == 'narrow':
            for idx in range(50):
                filename = FILENAME_TEMPLATE.format(idx)
                for idx_j in range(num_pods):
                    map = to_2d_array_level(filename)
                    play_trace = generate_play_trace_narrow(map, prob, rep, actions_list, render=render)
                    numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                    np.savez(f'expert_trajectories/wide/expert_zelda_{rep_as_str}_{idx}_{idx_j}', **numpy_dict)
        elif rep_as_str == 'turtle':
            for idx in range(50):
                filename = FILENAME_TEMPLATE.format(idx)
                for idx_j in range(num_pods):
                    map = to_2d_array_level(filename)
                    play_trace = generate_play_trace_turtle(map, prob, rep, actions_list, render=render)
                    numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                    np.savez(f'expert_trajectories/wide/expert_zelda_{rep_as_str}_{idx}_{idx_j}', **numpy_dict)
        else:
            sys.exit(1)
    else:
        filename = "zelda_lvl{}.txt".format(level)
        if rep_as_str == 'wide':
            for idx in range(num_pods):
                map = to_2d_array_level(filename)
                play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=render)
                numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                np.savez(f'expert_trajectories/wide/expert_zelda_{rep_as_str}_{level}_{idx}', **numpy_dict)
        elif rep_as_str == 'narrow':
            for idx in range(num_pods):
                map = to_2d_array_level(filename)
                play_trace = generate_play_trace_narrow(map, prob, rep, actions_list, render=render)
                numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                np.savez(f'expert_trajectories/wide/expert_zelda_{rep_as_str}_{idx}_{idx}', **numpy_dict)
        elif rep_as_str == 'turtle':
            for idx in range(num_pods):
                map = to_2d_array_level(filename)
                play_trace = generate_play_trace_turtle(map, prob, rep, actions_list, render=render)
                numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                np.savez(f'expert_trajectories/wide/expert_zelda_{rep_as_str}_{idx}_{idx}', **numpy_dict)


if __name__ == '__main__':
    main()
