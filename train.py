import argparse
from model import FullyConvPolicyBigMap, CustomPolicyBigMap

from utils import make_vec_envs

from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--exp_fn", help="Filename of the expert trajectory to use for training", type=str)
parser.add_argument("--rep", help="The representation type", type=str)
parser.add_argument("-s", "--from_scratch", help="Train a file from scratch", type=str)


def main ():
    args = parser.parse_args()
    exp_traj_fn = args.exp_fn
    rep_as_str = args.rep
    from_scratch = args.from_scratch
    from_scratch = True if from_scratch == 't' else False
    env_name = "zelda-wide-v0"
    log_dir = 'runs/wide'

    kwargs_dict = {'resume': False, 'cropped_size': 22, 'render_rank': 0, 'render': True}

    if rep_as_str == 'wide':
        policy = FullyConvPolicyBigMap
    else:
        policy = CustomPolicyBigMap

    env = make_vec_envs(env_name, rep_as_str, log_dir, n_cpu=1, **kwargs_dict)

    model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    if from_scratch:
        model.load(f'models/{rep_as_str}/zelda_{rep_as_str}', env=env)

    dataset = ExpertDataset(expert_path=f'expert_trajectories/{rep_as_str}/{exp_traj_fn}.npz', traj_limitation=-1,
                            batch_size=1)
    model.pretrain(dataset, n_epochs=10)
    model.save(f'models/{rep_as_str}/zelda_{rep_as_str}')


if __name__ == '__main__':
    main()
