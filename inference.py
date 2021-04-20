"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import pandas as pd

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO2.load(model_path, env=env)
    obs = env.reset()

    obs = env.reset()
    dones = False
    successful_levels = 0.0
    total_iterations = 0.0
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            total_iterations += 1
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                # print(info[0])
                pass
            if info[0]['solved']:
                successful_levels += 1
                dones = True
            if dones:
                break
    return successful_levels / total_iterations


################################## MAIN ########################################
game = 'zelda'
representation = 'wide'
model_path = f'models/{representation}/zelda_{representation}'

if __name__ == '__main__':
    change_percents = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    succ_lvls_per_chg_pct_val_pct = []
    vals_dict = {'chg_pct_vals': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'succ_lvls_pct': []}
    for change_percent in change_percents:
        kwargs = {
            'change_percentage': change_percent,
            'trials': 1000,
            'verbose': True,
            'render': False
        }
        pct_succ_levels = infer(game, representation, model_path, **kwargs)
        vals_dict['succ_lvls_pct'].append(pct_succ_levels)

    df = pd.DataFrame(vals_dict)
    print(df)



