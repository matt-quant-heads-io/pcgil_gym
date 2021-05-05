import sys
import argparse

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image

import numpy as np
import random

from utils import to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format


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
    for num_tc_action in range(20):
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
    #creates an array of tile coordinates, shuffles them to the order in which they are to be
    # modified, can change to work from left to right if necessary, following tree search paper
    # where these concepts were first introduced
    for row in map:
        for col in map:
            tile_coors.append((row, col))
    tile_coors = np.array(tile_coors)
    tile_coors = np.flip(tile_coors)
    #random.shuffle(tile_coors)

    #change from left-right, to bottom right-top left
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
        #actions.remove(old_tile_type) commented out because some will not change
        # 3) select an action from the list
        new_tile_type = random.choice(actions)
        # 4) update the map with the new tile type
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map
        #play_trace.append(transition_info_at_step)
        play_trace.insert(0, transition_info_at_step)

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
    play_trace = []
    old_map = map.copy()
    tile_coors = []
    for row in map:
        for col in map:
            tile_coors.append((row, col))
    tile_coors = np.array(tile_coors)
    # find the number of rows and columns to properly allow the tile to not go out of boundaries
    num_rows = tile_coors[len(tile_coors)-1][0]
    num_cols = tile_coors[len(tile_coors)-1][1]
    random.shuffle(tile_coors)
    starting_tile = tile_coors[8]
    row_idx, col_idx = starting_tile[0], starting_tile[1]
    moves = [0,1,2,3]

    for num_tc_action in range(20):
        new_map = old_map.copy()
        transition_info_at_step = [None, old_map, None, None]
        actions = actions_list.copy()
        new_map[row_idx] = old_map[row_idx].copy()
        transition_info_at_step[2] = (row_idx, col_idx)
        # 1) get the tile type at row_i, col_i and remove it from the list
        old_tile_type = map[row_idx][col_idx]
        transition_info_at_step[3] = old_tile_type
        # 2) remove the tile from the actions_list
        actions.remove(old_tile_type)
        # 3) select an action from the list
        new_tile_type = random.choice(actions)


        change_dir = False
        while(change_dir == False): # while loop implemented so there can be a turtle move in place
            next_move = random.randint(0, 3)
            if(next_move == 0 and row_idx != 0): # move left
                row_idx -= 1
                change_dir = True
            elif(next_move == 1 and row_idx != num_cols): # move right
                row_idx += 1
                change_dir = True
            elif(next_move == 2 and col_idx != 0): # move up
                col_idx -= 1
                change_dir = True
            elif(next_move == 3 and col_idx != num_rows): # move down
                col_idx += 1
                change_dir = True

        # 4) update the map with the new tile type
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map
        #play_trace.append(transition_info_at_step)
        play_trace.insert(0, transition_info_at_step)

        old_map = new_map

        # Render
        if render:
            map_img = render_map(new_map, prob, rep)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            input(f'')
            ren.close()
    return play_trace


def get_numpy_dict_from_play_trace_wide(play_trace, prob, rep):
    actions = []
    obs = []
    rewards = []
    episode_returns = []
    episode_starts = [True] + np.array([False]) * (len(play_trace) - 1)
    episode_return = 0.0
    rep.set_map(play_trace[0][1])
    for tuple_idx, pt_tuple in enumerate(play_trace):
        ob = convert_ob_to_int_arr(rep.get_observation())
        ob = int_map_to_onehot(ob)
        obs.append(ob)
        rep.set_map(pt_tuple[0])
        action = convert_action_to_npz_format(pt_tuple[2][0], pt_tuple[2][1], pt_tuple[-1], ob)
        actions.append(np.array([action]))
        reward = prob.get_reward(prob.get_stats(pt_tuple[0]), prob.get_stats(pt_tuple[1]))
        rewards.append(np.array(reward))
        episode_return += reward
    episode_returns.append(episode_return)
    actions = np.array(actions)
    obs = np.array(obs)
    rewards = np.array(rewards)
    episode_returns = np.array(episode_returns)
    episode_starts = np.array(episode_starts)

    return {
        'actions':  actions,
        'obs': obs,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

def get_numpy_dict_from_play_trace_narrow(play_trace, prob, rep):
    actions = []
    obs = []
    rewards = []
    episode_returns = []
    episode_starts = [True] + np.array([False]) * (len(play_trace) - 1)
    episode_return = 0.0
    rep.set_map(play_trace[0][1])
    for tuple_idx, pt_tuple in enumerate(play_trace):
        ob = convert_ob_to_int_arr(rep.get_observation())
        ob = int_map_to_onehot(ob)
        obs.append(ob)
        rep.set_map(pt_tuple[0])
        action = convert_action_to_npz_format(pt_tuple[2][0], pt_tuple[2][1], pt_tuple[-1], ob)
        actions.append(np.array([action]))
        reward = prob.get_reward(prob.get_stats(pt_tuple[0]), prob.get_stats(pt_tuple[1]))
        rewards.append(np.array(reward))
        episode_return += reward
    episode_returns.append(episode_return)
    actions = np.array(actions)
    obs = np.array(obs)
    rewards = np.array(rewards)
    episode_returns = np.array(episode_returns)
    episode_starts = np.array(episode_starts)

    return {
        'actions':  actions,
        'obs': obs,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

def get_numpy_dict_from_play_trace_turtle(play_trace, prob, rep):
    actions = []
    obs = []
    rewards = []
    episode_returns = []
    episode_starts = [True] + np.array([False]) * (len(play_trace) - 1)
    episode_return = 0.0
    rep.set_map(play_trace[0][1])
    for tuple_idx, pt_tuple in enumerate(play_trace):
        ob = convert_ob_to_int_arr(rep.get_observation())
        ob = int_map_to_onehot(ob)
        obs.append(ob)
        rep.set_map(pt_tuple[0])
        action = convert_action_to_npz_format(pt_tuple[2][0], pt_tuple[2][1], pt_tuple[-1], ob)
        actions.append(np.array([action]))
        reward = prob.get_reward(prob.get_stats(pt_tuple[0]), prob.get_stats(pt_tuple[1]))
        rewards.append(np.array(reward))
        episode_return += reward
    episode_returns.append(episode_return)
    actions = np.array(actions)
    obs = np.array(obs)
    rewards = np.array(rewards)
    episode_returns = np.array(episode_returns)
    episode_starts = np.array(episode_starts)

    return {
        'actions':  actions,
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
                    numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
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
