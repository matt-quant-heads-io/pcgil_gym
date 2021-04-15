import sys
import argparse
import numpy as np
import json
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image

import numpy as np
import random

from utils import str_hamming_distance_pct, to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format, string_from_2d_arr
from generate_pod_pt import generate_play_trace_wide, get_numpy_dict_from_play_trace_wide
import pika

# read queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
read_channel = connection.channel()
read_channel.exchange_declare(exchange='pod_generation_exchange', exchange_type='direct')


MIN_POD_END_STATES_SIZE = 30
LEVEL_INCREASE_SIZE = 10

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--levels_set", help="The levels set (1 to 5) to use in generating pts", type=int)
parser.add_argument("-r", "--render", help="Determines whether to render pod iterations", type=str, default='f')
parser.add_argument("--rep", help="The representation to use", type=str)
parser.add_argument("--dry_run", help="Determines whether to perform write operations (when set to t)", type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    levels_set = args.levels_set
    rep = args.rep
    render = True if args.render == 't' else False
    dry_run = True if args.dry_run == 't' else False

    level_hamm_report = f"level_hamm_reports/{rep}/level_hamm_report_{levels_set}.json"

    with open(level_hamm_report, 'r') as f:
        level_hamm_data = f.read()

    level_hamm_report_dict = json.loads(level_hamm_data)
    updated_level_hamm_report_dict = {k: v for k, v in level_hamm_report_dict.items()}

    prob = ZeldaProblem()
    rep = REPRESENTATIONS[rep]()
    actions_list = [act for act in list(TILES_MAP.values())]
    complete_numpy_dict = {'actions': [], 'obs': [], 'rewards': [],
                           'episode_returns': [], 'episode_starts': []}
    avg_hamm_for_good_level = []

    def callback(ch, method, properties, body):
        # Iterate through the keys to build out the level set per good level
        for good_level, pod_end_states in level_hamm_report_dict.items():
            if good_level == 'hamming_values':
                continue
            pod_end_states_updated = pod_end_states.copy()
            if len(pod_end_states_updated) < MIN_POD_END_STATES_SIZE:
                print(f"Length of pod_end_states < {MIN_POD_END_STATES_SIZE}, generating new levels")
                while len(pod_end_states_updated) < MIN_POD_END_STATES_SIZE:
                    map = to_2d_array_level(f"{good_level}.txt")
                    play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=render)
                    end_state = play_trace[-1][0]
                    end_state_str = string_from_2d_arr(end_state)
                    if end_state_str not in pod_end_states_updated:
                        print(f"Added new end state to level_hamm_report_{levels_set}")
                        pod_end_states_updated.append(end_state_str)
                        numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                        for k, v in numpy_dict.items():
                            complete_numpy_dict[k].extend(numpy_dict[k])
                updated_level_hamm_report_dict[good_level] = pod_end_states_updated
            else:
                # If size of pods list for good level is >= MIN_POD_END_STATES_SIZE then can start to add hamming heuristics
                levels_generated = 0
                while levels_generated < LEVEL_INCREASE_SIZE:
                    map = to_2d_array_level(f"{good_level}.txt")
                    play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=render)
                    end_state = play_trace[-1][0]
                    end_state_str = string_from_2d_arr(end_state)
                    if end_state_str not in pod_end_states_updated:
                        print(f"Added new end state to level_hamm_report_{levels_set}")
                        pod_end_states_updated.append(end_state_str)
                        numpy_dict = get_numpy_dict_from_play_trace_wide(play_trace, prob, rep)
                        for k, v in numpy_dict.items():
                            complete_numpy_dict[k].extend(numpy_dict[k])
                        levels_generated += 1
                updated_level_hamm_report_dict[good_level] = pod_end_states_updated

                # Now compute the avg. hamming value for updated_level_hamm_report_dict[good_level]
                # Do the following in a loop 10 times:
                avg_hamm_per_pop = []
                for i in range(10):
                    temp_level_set = updated_level_hamm_report_dict[good_level].copy()
                    ref_map = random.choice(temp_level_set)
                    temp_level_set.remove(ref_map)
                    population = random.choices(temp_level_set, k=20)
                    for comp_map in population:
                        avg_hamm_per_pop.append(str_hamming_distance_pct(ref_map, comp_map))
                avg_hamm_val = sum(avg_hamm_per_pop) / len(avg_hamm_per_pop)
                avg_hamm_for_good_level.append(avg_hamm_val)
                print(f"computed avg hamming for {good_level}: {avg_hamm_val}")

        if not dry_run:
            np.savez(f'expert_trajectories/wide/expert_zelda_complete_{levels_set}', **complete_numpy_dict)
            print(f"Saved expert trajectory as expert_zelda_complete_{levels_set}")
            if len(avg_hamm_for_good_level) > 0:
                print(f"Updated avg hamming for level_set {levels_set}: {avg_hamm_for_good_level}")
                updated_level_hamm_report_dict["hamming_values"] = avg_hamm_for_good_level

            with open(level_hamm_report, 'w') as f:
                json.dump(updated_level_hamm_report_dict, f, indent=4)
                print(f"Updated level_hamm_report_{levels_set}")

    read_channel.queue_declare(queue="level_set_{}".format(levels_set), exclusive=False)
    read_channel.queue_bind(exchange='pod_generation_exchange', queue="level_set_{}".format(levels_set),
                            routing_key="level_set_{}".format(levels_set))

    read_channel.basic_consume(queue="level_set_{}".format(levels_set), on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for pod generation trigger. To exit press CTRL+C')

    read_channel.start_consuming()




