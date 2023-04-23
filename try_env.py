import gym
import numpy as np
import pandas as pd

import sinergym
from sinergym.utils.wrappers import LoggerWrapper

extra_conf={
    'timesteps_per_hour':4, 
    'runperiod':(1,6,2021,1,9,2021), # (d, m, y)
}

weather_file = 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'

env = gym.make('Eplus-demo-v1',
               weather_file=weather_file,
               config_params=extra_conf,)


data_buffer = pd.DataFrame(columns=['zone temperature', 
                                    'Site Outdoor Air Drybulb Temperature(Environment)',
                                    'Site Outdoor Air Relative Humidity(Environment)',
                                    'Site Wind Speed(Environment)',
                                    'Site Total Solar Radiation Rate per Area(Environment)',
                                    'Zone People Occupant Count(SPACE1-1)',
                                    'control signal',])

weather_data = []

for i in range(1):
    obs = env.reset()
    rewards = []
    done = False
    current_month = 0
    while not done:

        a = env.action_space.sample()

        obs, reward, done, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()

# print(weather)

# save data buffer to csv
# data_buffer.to_csv('wdata_buffer_pittsburg_Jun1Sept1_random_action.csv')

# print the sum of rewards of first 2000 time steps
print(sum(rewards[:2000]))

# read zimages/reward_plus.csv
reward_plus = pd.read_csv('zimages/reward_plus.csv', index_col=0)

# print the sum of rewards of first 2000 time steps
print(sum(reward_plus['reward'][:2000]))
