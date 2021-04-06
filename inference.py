import gym
import gym_pcgrl

# assume input is from generated training eps via matt



###### determine unsuccessful/successful
# env = gym.make('sokoban-narrow-v0')
agents = ['zelda-wide-v0', 'zelda-narrow-v0', 'zelda-turtle-v0']
env = gym.make(agents[0]) # use zelda-wide-v0
obs = env.reset()
trackingstruct = {}
'''
{
  agent_name: [{
    change_percentage: 0.1,
    success_ct: 4,
    failure_ct: 5,
    total_runs: 9
  }]
}
'''

# [todo] clarify below values to use in change percenage (in inference.py)
vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for val in vals:
  env.kwargs = { # set new kwargs
      'change_percentage': val, # to use values of (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
      'trials': 40, # 40 trials
      'verbose': True
      }
  #########################################

  ###### determine unsuccessful/successful
  # solver code for 1 trial
  success = False
  for t in range(1000):

    # pick action here, create an [change, x, y] action using new agent
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    env.render('human')
    if done: # mark as success
      print(info) # {'player': 3, 'key': 2, 'door': 2, 'enemies': 7, 'regions': 5, 'nearest-enemy': 0, 'path-length': 0, 'iterations': 17, 'changes': 15, 'max_iterations': 1155, 'max_changes': 15}
      # % tile-change actions: info.changes
      print("Episode finished after {} timesteps".format(t+1))
      success = True
      break
  # if reach here, unsuccessful
  if not success:
    print('unsuccessful operation')
  #########################################

##### stats generation
# generate matplotlib graph for above (bullet 1) - to confirm, bullet1 merges into the last line of bullet2
# bullet2 is just all of the agents separately with the same data

## actionables
# generate avg % tile-change number
# generate matplotlib per agent[zelda-wide-v0, zelda-narrow-v0...], x=change_percentage,y=avg%tile_change
#########################################


##### list out possible agents to use from pcgrl
from gym import envs
import gym_pcgrl

# print(sorted([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point]))
# 'zelda-narrow-v0', 'zelda-narrowcast-v0', 'zelda-narrowmulti-v0', 'zelda-turtle-v0', 'zelda-turtlecast-v0', 'zelda-wide-v0'
# '''