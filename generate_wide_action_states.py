# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import argparse
import copy

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from generate_pod_pt import generate_play_trace_narrow, generate_play_trace_wide, generate_play_trace_turtle

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image

import numpy as np
import random
INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7}
from utils import to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format, to_action_space

def to_char_level(map, map_path):
    #add border back
    map_copy = map.copy()
    key_list = list(TILES_MAP.keys())
    val_list = list(TILES_MAP.values())

    for row in range(len(map_copy)):
        for col in range(len(map_copy[row])):
            pos = val_list.index(map_copy[row][col])
            map_copy[row][col] = key_list[pos]
    arr = np.array(map_copy)

    num_cols = arr.shape[1]
    border_rows = []
    border_cols = []
    for i in range(num_cols):
        border_rows.append('w')

    border_rows = np.array(border_rows)
    border_rows = border_rows.reshape((1, num_cols))
    arr = np.vstack((border_rows, arr))
    arr = np.vstack((arr, border_rows))
    # print(arr)

    num_rows = arr.shape[0]
    for i in range(num_rows):
        border_cols.append('w')

    border_cols = np.array(border_cols)
    border_cols = border_cols.reshape((num_rows, 1))
    # print(border_cols.shape)
    arr = np.hstack((border_cols, arr))
    arr = np.hstack((arr, border_cols))
    return arr


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# file_path = 'good_levels/Zelda/zelda_lvl{}.txt'
rep = 'wide'
prob = ZeldaProblem()
actions_list = [act for act in list(TILES_MAP.values())]
for idx in range(50):
    map = to_2d_array_level('good_levels/Zelda/zelda_lvl{}.txt'.format(idx))
    im = prob.render(map)
    im.save('good_levels_set/good_levels/Zelda/zelda_lvl{}.png'.format(idx),
            "PNG")
    pt = []
    for init_map_idx in range(30):
        # print(map)
        use_map = map
        action_space = []
        play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=False)
        init_action_space = 'expert_trajectories_{}_orderlist/init_maps_level{}/action_space_{}.txt'.format(rep, idx,
                                                                                                        init_map_idx)
        for pt in play_trace:
            action_space.append([pt[-2][0], pt[-2][1], INT_MAP[pt[-1]]])
        action_space = np.array(action_space)
        np.savetxt(init_action_space, action_space, delimiter = ' ', fmt='%s')
        init_map = play_trace[0][0]
        init_map_path = 'expert_trajectories_{}_orderlist/init_maps_level{}/destroyed_map_{}.txt'.format(rep, idx,
                                                                            init_map_idx)
        # print(init_map_path)
        arr = to_char_level(copy.deepcopy(init_map), init_map_path)
        # arr = to_char_level(init_map.deepcopy(), init_map_path)
        np.savetxt(init_map_path, arr, delimiter='', fmt='%s')
        im = prob.render(init_map)
        im.save('expert_trajectories_{}_orderlist/init_maps_level{}/destroyed_map_{}.png'.format(rep, idx, init_map_idx), "PNG")



        # for pt in range(1, len(play_trace)):
        #     first_x = play_trace[pt-1][-2][0]
        #     first_y = play_trace[pt-1][-2][1]
        #     sec_x = play_trace[pt][-2][0]
        #     sec_y = play_trace[pt][-2][1]
        #     tile_type = play_trace[pt][-1]
        #     if(sec_x - first_x == 1):
        #         action_space.append('right')
        #     elif(sec_x - first_x == -1):
        #         action_space.append('left')
        #     elif(sec_y - first_y == 1):
        #         action_space.append('down')
        #     else:
        #         action_space.append('up')
        #     action_space.append(tile_type)
#         #     For Narrow
#         # for pt in play_trace:
#         #     tile_x = pt[-2][0]
#         #     tile_y = pt[-2][1]
#         #     start_tile = pt[1][tile_x][tile_y]
#         #     new_tile = pt[0][tile_x][tile_y]
#         #     # print([start_tile, new_tile])
#         #     if(start_tile == new_tile):
#         #         action_space.append('no-action')
#         #     else:
#         #         action_space.append(new_tile)

#         goal_map = play_trace[-1][1]
#         init_map = play_trace[0][0]
#         print(action_space)
#         init_map_path = 'expert_trajectories_{}/init_maps_level{}/starting_map_{}.txt'.format(rep, idx, init_map_idx)
#         # to_char_level(init_map, init_map_path)
#         init_action_space = 'expert_trajectories_{}/init_maps_level{}/action_space_{}.txt'.format(rep, idx, init_map_idx)
#         action_space = np.array(action_space)
#         np.savetxt(init_action_space, action_space, delimiter = ' ', fmt='%s')
        # print(prob.get_stats(init_map_path))
# print(action_space)


# init_map = random.randint(0, 50)
# starting_map = random.randint(0, 30)
# print(init_map, starting_map)
# for init_map in range(50):
#     for starting_map in range(30):
#         start_map = to_2d_array_level('expert_trajectories_turtle/init_maps_level{}/starting_map_{}.txt'.format(init_map, starting_map))
#         # action_space = to_action_space('expert_trajectories_wide/init_maps_level{}/action_space_{}.txt'.format(init_map, starting_map))
#         im = prob.render(start_map)
#         im.save('expert_trajectories_turtle/init_maps_level{}/starting_map_{}.png'.format(init_map, starting_map), "PNG")

# init_map = to_2d_array_level('expert_trajectories_wide/init_maps_level{}/init_map_{}.txt'.format(init_map, starting_map))

# start_map = to_2d_array_level('good_levels/Zelda/zelda_lvl0.txt')
# # im = prob.render(start_map)
# # im.save('expert_trajectories_wide/init_maps_level0/orig_map.png')
# for row in range(len(start_map)):
#     for col in range(len(start_map[row])):
#         print([row, col], start_map[row][col])
#
# play_trace = generate_play_trace_narrow(start_map, prob, 'narrow', actions_list, render=False)
# init_map = play_trace[-1][1]
# im = prob.render(init_map)
# im.save('expert_trajectories_narrow/init_maps_level0/starting_map_0.png', "PNG")

