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

        # add entry to data buffer
        data_buffer = data_buffer.append({'zone temperature': obs['Zone Air Temperature(SPACE1-1)'],
                                            'Site Outdoor Air Drybulb Temperature(Environment)': obs['Site Outdoor Air Drybulb Temperature(Environment)'],
                                            'Site Outdoor Air Relative Humidity(Environment)': obs['Site Outdoor Air Relative Humidity(Environment)'],
                                            'Site Wind Speed(Environment)': obs['Site Wind Speed(Environment)'],
                                            'Site Total Solar Radiation Rate per Area(Environment)': obs['Site Diffuse Solar Radiation Rate per Area(Environment)']+obs['Site Direct Solar Radiation Rate per Area(Environment)'],
                                            'Zone People Occupant Count(SPACE1-1)': obs['Zone People Occupant Count(SPACE1-1)'],
                                            'control signal': a}, ignore_index=True)


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
data_buffer.to_csv('wdata_buffer_pittsburg_Jun1Sept1_random_action.csv')
