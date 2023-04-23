from typing import List, Any, Sequence
from sinergym.utils.common import get_season_comfort_range
from datetime import datetime
import gym
import numpy as np
import sinergym

extra_conf={
    'timesteps_per_hour':4, 
    'runperiod':(1,6,2021,1,9,2021), # (d, m, y)
}

weather_file = 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'

env = gym.make('Eplus-demo-v1',
               weather_file=weather_file,
               config_params=extra_conf,)


from sinergym.utils.controllers import RBC5Zone

class MyRuleBasedController(RBC5Zone):

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on outdoor air drybulb temperature and daytime.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = observation

        out_temp = obs_dict['Site Outdoor Air Drybulb Temperature(Environment)']

        day = int(obs_dict['day'])
        month = int(obs_dict['month'])
        hour = int(obs_dict['hour'])
        year = int(obs_dict['year'])

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            season_comfort_range = self.setpoints_summer
        else:
            season_comfort_range = self.setpoints_summer
        season_comfort_range = get_season_comfort_range(1991,month, day)
        # Update setpoints
        in_temp = obs_dict['Zone Air Temperature(SPACE1-1)']

        current_heat_setpoint = obs_dict[
            'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)']
        current_cool_setpoint = obs_dict[
            'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if in_temp < season_comfort_range[0]:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif in_temp > season_comfort_range[1]:
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        action = (new_heat_setpoint, new_cool_setpoint)
        if current_dt.weekday() > 5 or hour in range(22, 6):
            #weekend or night
            action = (18.33, 23.33)

        return action

COMFORT_RANGE = (23, 26)
    
def reward_(obs):
    comfort_range = COMFORT_RANGE
    weight_energy = 1

    if obs['Zone People Occupant Count(SPACE1-1)'] > 0:
        weight_energy = 0.1
    
    x = obs['Zone Air Temperature(SPACE1-1)']

    comfort_cost = (1 - weight_energy) * (abs(x - comfort_range[0]) + abs(x - comfort_range[1]))
    energy_cost = weight_energy * obs['Facility Total HVAC Electricity Demand Rate(Whole Building)'] / 100

    return - comfort_cost - energy_cost

# create rule-based controller
agent = MyRuleBasedController(env)

for i in range(1):
    obs = env.reset()
    rewards = []
    done = False
    current_month = 0
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward_(obs))
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

import pandas as pd
# print the sum of rewards of first 2000 time steps
print(sum(rewards[:2000]))

# read zimages/reward_plus.csv
reward_plus = pd.read_csv('zimages/reward_plus.csv', index_col=0)

# print the sum of rewards of first 2000 time steps
print(sum(reward_plus['reward'][:2000]))