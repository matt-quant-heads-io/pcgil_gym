import gym
import gym_pcgrl

###### assume input is from generated training eps via matt

###### determine unsuccessful/successful
agents = ['zelda-wide-v0', 'zelda-narrow-v0', 'zelda-turtle-v0']
# agents = ['zelda-wide-v0']
tracking = {}

for agent_type in agents:
  tracking[agent_type] = []

  # env = gym.make(agents[0]) # use zelda-wide-v0 for testing
  env = gym.make(agent_type)
  obs = env.reset()

  # [todo] confirm vals
  vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  for val in vals:
    per_change_percentage = {
      'change_percentage': val,
      'tile_change_actions': None # remain None if false
    }
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
        # print(info) # {'player': 3, 'key': 2, 'door': 2, 'enemies': 7, 'regions': 5, 'nearest-enemy': 0, 'path-length': 0, 'iterations': 17, 'changes': 15, 'max_iterations': 1155, 'max_changes': 15}
        # % tile-change actions: info.changes
        # print("Episode finished after {} timesteps".format(t+1))
        success = True
        per_change_percentage['tile_change_actions'] = info['changes']
        break
    # if reach here, unsuccessful
    if not success:
      print('unsuccessful operation')

    tracking[agent_type].append(per_change_percentage)

import json # read result data struct
print(json.dumps(tracking, indent=2))

##### stats generation
# generate matplotlib graph for above (bullet 1) - to confirm, bullet1 merges into the last line of bullet2
# bullet2 is just all of the agents separately with the same data

## actionables
# generate avg % tile-change number (1)
# generate matplotlib per agent[zelda-wide-v0, zelda-narrow-v0...], x=change_percentage,y=avg%tile_change (2)
import matplotlib.pyplot as plt
import numpy as np
ys = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for agent in agents: 
  base = tracking[agent]
  result = [elem['tile_change_actions'] for elem in base]
  print('average % tile-change number for ' + agent + ': ', np.mean(result)) # (1)
  plt.plot(result, ys)
plt.savefig('tileactions_vs_changepercentage.png') # (2)
plt.close()
#########################################

##### list out possible agents to use from pcgrl
# 'zelda-narrow-v0', 'zelda-narrowcast-v0', 'zelda-narrowmulti-v0', 'zelda-turtle-v0', 'zelda-turtlecast-v0', 'zelda-wide-v0'
