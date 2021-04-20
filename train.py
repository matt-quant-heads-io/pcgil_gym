import argparse
from model import FullyConvPolicyBigMap, CustomPolicyBigMap

from utils import make_vec_envs

from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2
import time


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--exp_fn", help="Filename of the expert trajectory to use for training", type=str)
parser.add_argument("--rep", help="The representation type", type=str)
parser.add_argument("-s", "--from_scratch", help="Train a file from scratch", type=str)


# def callback(_locals, _globals):
#     """
#     Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
#     :param _locals: (dict)
#     :param _globals: (dict)
#     """
#     global n_steps, best_mean_reward
#     # Print stats every 1000 calls
#     if (n_steps + 1) % 10 == 0:
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#         print(f"len(x) is {len(x)}")
#         if len(x) > 100:
#            #pdb.set_trace()
#             mean_reward = np.mean(y[-100:])
#             print(x[-1], 'timesteps')
#             print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
#
#             # New best model, we save the agent here
#             if mean_reward > best_mean_reward:
#                 best_mean_reward = mean_reward
#                 # Example for saving best model
#                 print("Saving new best model")
#                 print(f"_locals['self'] is {_locals['self']}")
#                 _locals['self'].model.save(os.path.join(log_dir, 'best_model.pkl'))
#             else:
#                 print("Saving latest model")
#                 print(f"_locals['self'] is {_locals['self']}")
#                 _locals['self'].model.save(os.path.join(log_dir, 'latest_model.pkl'))
#         else:
#             print('{} monitor entries'.format(len(x)))
#             pass
#     n_steps += 1
#     # Returning False will stop training early
#     return True


def main(exp_traj_fn, rep_as_str, from_scratch):
    env_name = f"zelda-{rep_as_str}-v0"
    log_dir = f'runs/{rep_as_str}'

    kwargs_dict = {'resume': False, 'render': True}

    if rep_as_str == 'wide':
        policy = FullyConvPolicyBigMap
    else:
        policy = CustomPolicyBigMap

    env = make_vec_envs(env_name, rep_as_str, log_dir, n_cpu=1, **kwargs_dict)

    model = PPO2(policy, env, verbose=1, tensorboard_log=f"./runs/{rep_as_str}")
    if not from_scratch:
        model.load(f'models/{rep_as_str}/zelda_{rep_as_str}', env=env)

    dataset = ExpertDataset(expert_path=f'expert_trajectories/{rep_as_str}/{exp_traj_fn}.npz', traj_limitation=-1,
                            batch_size=15)
    start_time = time.process_time()
    model.set_env(env)
    model.pretrain(dataset, n_epochs=15)
    end_time = time.process_time()
    print(f"training took {end_time - start_time} seconds")
    model.save(f'models/{rep_as_str}/zelda_{rep_as_str}')


if __name__ == '__main__':
    args = parser.parse_args()
    exp_traj_fn = args.exp_fn
    rep_as_str = args.rep
    from_scratch = args.from_scratch
    from_scratch = True if from_scratch == 't' else False
    main(exp_traj_fn, rep_as_str, from_scratch)
