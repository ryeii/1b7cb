import numpy as np
import pandas as pd
from datetime import datetime
import json

'''
Convention:
- x (numpy array): zone state + environment state, shape (state_dim,)
    - x[0]: zone temperature
    - x[1]: outdoor air drybulb temperature
    - x[2]: outdoor air relative humidity
    - x[3]: site wind speed
    - x[4]: site total solar radiation rate per area
    - x[5]: zone occupant count
- u (float): the control signal, int in [0, 9], shape (control_dim,)
- t (int): the current time
'''

COMFORT_RANGE = (23, 26)
WEATHER = pd.read_csv('weather_pittsburg_Jun1Sept1.csv')
LAMBDA_CONFIDENCE = 1.0

from typing import List, Any, Sequence
from sinergym.utils.common import get_season_comfort_range
import sinergym
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


def env_reader(timestep):
    '''
    Return the environment state at the given timestep.

    Args:
    - timestep (int): timestep to read the environment state from
    - weather_df (pandas dataframe): the weather dataframe

    Returns:
    - env_state (numpy array): the environment state at the given timestep, shape (1,)
    '''
    return WEATHER.iloc[timestep].values.tolist()


def get_confidence_value(var):
    '''
    Return the confidence value given the variance.

    Args:
    - var (float): the variance of the Gaussian Process prediction

    Returns:
    - confidence_value (float): the confidence value
    '''
    return 0.5 * np.log(2 * np.pi * var) + 0.5 # information entropy of Gaussian is 1/2 * log(2*pi*var) + 1/2


def dynamics(model, standardization, x, u, t):
    '''
    Predict the next zone temperature given the current state, control signal, and time.

    Args:
    - x (ndarray): a vector of zone states (zone temperature), shape (state_dim,)
    - u (ndarray): a vector of perturbated control signals, shape (control_dim,)
    - t (float): the current time

    Returns:
    - mu (float): the mean of the Gaussian Process prediction
    - var (float): the variance of the Gaussian Process prediction
    '''

    # convert x and u to standardized form
    x = (x - standardization['zone temperature']['mean']) / standardization['zone temperature']['std']
    u = (u - standardization['control signal']['mean']) / standardization['control signal']['std']

    env_state = env_reader(t)

    env_state_name = ['Site Outdoor Air Drybulb Temperature(Environment)',
                    'Site Outdoor Air Relative Humidity(Environment)',
                    'Site Wind Speed(Environment)',
                    'Site Total Solar Radiation Rate per Area(Environment)',
                    'Zone People Occupant Count(SPACE1-1)',]

    # convert env_state to standardized form
    for variable in range(5):
        if standardization[env_state_name[variable]]['std'] != 0:
            env_state[variable] = (env_state[variable] - standardization[env_state_name[variable]]['mean']) / standardization[env_state_name[variable]]['std']


    '''
    sometimes all values in the buffer for this variable are zeros, which will cause the std to be zero and standardized form to be nan due to division by zero.
    In this case, we set the standardized form to be 0.
    '''
    # replace nan in env_state with 0
    env_state = np.nan_to_num(env_state)

    # make an array of x with shape (len(x), 7)
    x_array = np.zeros((len(x), 7))

    # replace each row in x_array with [x, env_state[0], env_state[1], env_state[2], env_state[3], env_state[4], u]. This is the input to the GP model.
    for i in range(len(x)):
        x_array[i] = [x[i], env_state[0], env_state[1], env_state[2], env_state[3], env_state[4], u[i]]

    x = x_array

    # tensorize x
    x = torch.tensor(x, dtype=torch.float32)

    # reshape x so it has size 7 in dimension 0
    x = x.reshape(len(x), 7)

    pred = model(x)

    mu = pred.mean
    var = pred.variance

    # convert mu and var from tensor to numpy arrays
    mu = mu.detach().numpy()
    var = var.detach().numpy()

    # convert mu back to original form
    mu = mu * standardization['zone temperature']['std'] + standardization['zone temperature']['mean']

    return mu, var

def cost(x, var, u, t):
    '''
    Compute the cost of the given state and control signal.

    Args:
    - x (float): zone state (zone temperature), shape (state_dim,
    - sigma (float): standard deviation of the Gaussian Process prediction
    - u (float): the control signal, shape (control_dim,)

    Returns:
    - c (float): the cost of the given state and control signal
    '''
    env_state = env_reader(t)

    comfort_range = COMFORT_RANGE
    weight_energy = 1

    if env_state[4] > 0:
        weight_energy = 0.001

    comfort_cost = (1 - weight_energy) * (abs(x - comfort_range[0]) + abs(x - comfort_range[1]))
    energy_cost = weight_energy * (u - x)**2

    confidence = get_confidence_value(var)

    return comfort_cost + energy_cost - LAMBDA_CONFIDENCE * confidence

import gpytorch
import torch

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def fit(data_buffer):
    '''
    Args:
    - data_buffer (pandas DataFrame): the data buffer

    Returns:
    - model (GPModel): the trained Gaussian Process model
    - likelihood (gpytorch.likelihoods): the likelihood of the Gaussian Process model
    - standardization (dict): the mean and standard deviation of each column in the data buffer
    '''

    data = data_buffer.copy()

    # for each column, calculate and record the mean and standard deviation, then normalize the data
    standardization = {}
    for col in data.columns:
        standardization[col] = {'mean': data[col].mean(), 'std': data[col].std()}
        if data[col].std() != 0:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        else:
            data[col] = 0

    X = data
    Y = data['zone temperature'].shift(-1) # shift the zone temperature column up by 1 row

    # drop the last row of X and Y, since the last row of X has no corresponding Y
    X = X.drop(X.index[-1])
    Y = Y.drop(Y.index[-1])

    # X and Y to numpy arrays type float32
    X = X.to_numpy(dtype=np.float32)
    Y = Y.to_numpy(dtype=np.float32)

    # nan to 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if np.isnan(X[i][j]):
                X[i][j] = 0.0
    Y = np.nan_to_num(Y)

    # tensorize X and Y
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X, Y, likelihood)

    model.covar_module.base_kernel.lengthscale = 0.836
    model.covar_module.outputscale = 0.652
    model.likelihood.noise = 0.010

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 10

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood, standardization

class MPPIController:
    def __init__(self, num_samples, horizon, time_offset, dynamics_fn, cost_fn, data_buffer, lambda_=1.0, sigma=1.0):
        self.num_samples = num_samples
        self.horizon = horizon
        self.time_offset = time_offset
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.data_buffer = data_buffer
        self.lambda_ = lambda_
        self.sigma = sigma
        
    def control(self, x0, t0):
        """
        Compute the MPPI control signal given the current state and time.

        Args:
        - x0 (numpy array): the current state, shape (state_dim,)
        - t (float): the current time

        Returns:
        - u (numpy array): the control signal, shape (control_dim,)
        """
        # if the length of the data buffer is larger than 2, fit a Gaussian Process model to the data buffer

        if len(self.data_buffer) > 2:
                model, likelihood, standardization = fit(self.data_buffer)
        else:
            return False

        S = np.zeros((self.num_samples, self.horizon)) # sample trajectories
        V = np.zeros((self.num_samples, self.horizon)) # confidence trajectories
        C = np.zeros((self.num_samples,)) # trajectory costs
        U = np.zeros((self.horizon,)) # control signal
        U_noise = np.zeros((self.num_samples, self.horizon)) # noise added to U

        # populate U with random integers
        for j in range(self.horizon):
            U[j] = np.random.randint(24, 26)

        x = np.copy(x0)
        x = np.array([x for i in range(self.num_samples)])
        for t in range(self.horizon):
            s = np.random.normal(0, self.sigma, (self.num_samples,)) # sample noise
            U_noise[:, t] = s
            u = s + U[t] # get a vector of control signals by adding noise vector to initial U
            x, var = self.dynamics_fn(model, standardization, x, u, t0 + t) # pass t so it can call env_reader to get weather
            S[:, t] = x
            V[:, t] = var
        
        for i in range(self.num_samples):
            for j in range(self.horizon):
                C[i] += self.cost_fn(S[i, j], V[i, j], U[j], t0 + j)
        
        # for i in range(self.num_samples):
        #     x = np.copy(x0)
        #     s = np.random.normal(0, self.sigma, (self.horizon,)) # sample noise
        #     U_noise[i] = s
        #     for j in range(self.horizon):
        #         u = U[j] + s[j]

        #         x, var = self.dynamics_fn(model, standardization, x, u, self.time_offset + t) # pass t so it can call env_reader to get weather

        #         S[i, j] = x
        #         C[i] += self.cost_fn(x, var, u, self.time_offset + t) # occupancy is obtained from the env state read from weather file, so we don't need to pass t
        
        # downscale C by 1000 to avoid float underflow
        C /= 10000

        expC = np.exp(-self.lambda_ * C)

        expC /= np.sum(expC)
        
        for j in range(self.horizon):
            U[j] += np.sum(expC * U_noise[:, j])

        u = U[0]

        # '''
        # Fall back mechanism.
        # Using the dynamics function to evaluate the actions (u) chosen by MPPI and record the sum of the variances.
        # If the sum of the variances is larger than epsilon, then call rule-based controller and return the control signal.
        # '''
        # epsilon = 200
        # x = np.copy(x0)
        # vars = []
        # for j in range(self.horizon):
        #     x, var = self.dynamics_fn(model, standardization, [x], [U[j]], t0 + t)
        #     # vars.append(get_confidence_value(var[0]))
        #     vars.append(var[0])
        #     x = x[0]
        # # print('action sequence: ', U , 'vars: ', vars, 'vars_sum: ', sum(vars), 'data buffer length: ', len(self.data_buffer), 'timestep: ', t0)
        # if sum(vars) > epsilon:
        #     # return False
        #     u = False
        # # print('u: ', u)

        return u
    
    def update_data_buffer(self, data_buffer):
        self.data_buffer = data_buffer
        return



data_buffer = pd.DataFrame(columns=['zone temperature', 
                                    'Site Outdoor Air Drybulb Temperature(Environment)',
                                    'Site Outdoor Air Relative Humidity(Environment)',
                                    'Site Wind Speed(Environment)',
                                    'Site Total Solar Radiation Rate per Area(Environment)',
                                    'Zone People Occupant Count(SPACE1-1)',
                                    'control signal',])


import gym
import sinergym
from sinergym.utils.wrappers import LoggerWrapper

def reward_(obs):
    comfort_range = COMFORT_RANGE
    weight_energy = 1

    if obs['Zone People Occupant Count(SPACE1-1)'] > 0:
        weight_energy = 0.1
    
    x = obs['Zone Air Temperature(SPACE1-1)']

    comfort_cost = (1 - weight_energy) * (abs(x - comfort_range[0]) + abs(x - comfort_range[1]))
    energy_cost = weight_energy * obs['Facility Total HVAC Electricity Demand Rate(Whole Building)'] / 100

    return - comfort_cost, - energy_cost

extra_conf={
    'timesteps_per_hour':4, 
    'runperiod':(1,6,2021,1,9,2021), # (d, m, y)
}

weather_file = 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'

env = gym.make('Eplus-demo-v1',
               weather_file=weather_file,
               config_params=extra_conf,)

obs = env.reset()
data = []
done = False
current_timestep = 0
mppi = MPPIController(num_samples=100, horizon=4, time_offset=0, dynamics_fn=dynamics, cost_fn=cost, data_buffer=data_buffer, lambda_=1.0, sigma=5)

# start_time = datetime.datetime.now()

# create rule-based controller
rule_agent = MyRuleBasedController(env)

while not done:

    a = mppi.control(obs['Zone Air Temperature(SPACE1-1)'], current_timestep)
    a = (a, a)

    if a == (False, False):
        a = rule_agent.act(obs)
        a = (a[0], a[0])

    obs, reward, done, info = env.step(a)
    comfort_reward, energy_reward = reward_(obs)
    indoor_temp = obs['Zone Air Temperature(SPACE1-1)']
    action = (obs['Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)'])
    data.append([comfort_reward, energy_reward, indoor_temp, action])

    # add entry to data buffer using pd.concat
    data_buffer = pd.concat([data_buffer, pd.DataFrame({'zone temperature': obs['Zone Air Temperature(SPACE1-1)'],
                                        'Site Outdoor Air Drybulb Temperature(Environment)': obs['Site Outdoor Air Drybulb Temperature(Environment)'],
                                        'Site Outdoor Air Relative Humidity(Environment)': obs['Site Outdoor Air Relative Humidity(Environment)'],
                                        'Site Wind Speed(Environment)': obs['Site Wind Speed(Environment)'],
                                        'Site Total Solar Radiation Rate per Area(Environment)': obs['Site Diffuse Solar Radiation Rate per Area(Environment)']+obs['Site Direct Solar Radiation Rate per Area(Environment)'],
                                        'Zone People Occupant Count(SPACE1-1)': obs['Zone People Occupant Count(SPACE1-1)'],
                                        'control signal': a[0]}, index=[0])], ignore_index=True) # since summer, only the cooling setpoint is used
    
    mppi.update_data_buffer(data_buffer)

    current_timestep += 1

    if current_timestep % 100 == 0:
        # checkpoint_time = datetime.datetime.now()
        # predicted_total_time = (checkpoint_time - start_time) / current_timestep * 4 * 24 * 92
        # predicted_remaining_time = predicted_total_time - (checkpoint_time - start_time)
        # predicted_remaining_time_minutes = predicted_remaining_time.seconds / 60
        # print('timestep: ', current_timestep, '. time elapsed: ', checkpoint_time - start_time, '. predicted remaining time: ', predicted_remaining_time_minutes, ' minutes')
        print('timestep: ', current_timestep)

        # make data a pandas dataframe and save to csv
        pd.DataFrame(data, columns=['comfort_reward', 'energy_reward', 'indoor_temp', 'action']).to_csv('zimages/data_gp_exp1.csv', index=False)
        print('data saved!')
    
    if current_timestep == 2001:
        done = True

env.close()

# plot the reward, x-axis is the timestep and y-axis is the reward
import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()
plt.savefig('zimages/reward_plus.png')
