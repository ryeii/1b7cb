import numpy as np
import pandas as pd
import datetime
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
    # u is a float from 0 to 1, so we need to convert it to an int from 0 to 9
    for i in range(len(u)):
        u[i] = int(u[i] * 9)

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

    pred = pred.detach().numpy()

    # convert mu back to original form
    pred = pred * standardization['zone temperature']['std'] + standardization['zone temperature']['mean']

    return pred

def cost(x, u, t):
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
        weight_energy = 0.1

    comfort_cost = (1 - weight_energy) * (abs(x - comfort_range[0]) + abs(x - comfort_range[1]))
    energy_cost = weight_energy * (u - x)**2

    return comfort_cost + energy_cost

import torch

'''define neural network'''
class NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.silu(self.linear1(x))
        x = torch.nn.functional.silu(self.linear2(x))
        x = self.linear3(x)
        return x
    

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
        
    def control(self, x0, t):
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
            return np.random.randint(0, 10) # return random int from 0 to 9

        S = np.zeros((self.num_samples, self.horizon)) # sample trajectories
        C = np.zeros((self.num_samples,)) # trajectory costs
        U = np.zeros((self.horizon,)) # control signal
        U_noise = np.zeros((self.num_samples, self.horizon)) # noise added to U

        # populate U with random signals from 0 to 1
        for j in range(self.horizon):
            U[j] = np.random.uniform(0, 1)

        x = np.copy(x0)
        x = np.array([x for i in range(self.num_samples)])
        for t in range(self.horizon):
            s = np.random.normal(0, self.sigma, (self.num_samples,)) # sample noise
            U_noise[:, t] = s
            u = s + U[t] # get a vector of control signals by adding noise vector to initial U
            x = self.dynamics_fn(model, standardization, x, u, self.time_offset + t) # pass t so it can call env_reader to get weather
            S[:, t] = x
        
        for i in range(self.num_samples):
            for j in range(self.horizon):
                C[i] += self.cost_fn(S[i, j], U[j], self.time_offset + j)
        
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
        # u is a float from 0 to 1, so we need to convert it to an int from 0 to 9
        u = int(u * 9)
        
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

    return - comfort_cost - energy_cost

extra_conf={
    'timesteps_per_hour':4, 
    'runperiod':(1,6,2021,1,9,2021), # (d, m, y)
}

weather_file = 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'

env = gym.make('Eplus-demo-v1',
               weather_file=weather_file,
               config_params=extra_conf,)

obs = env.reset()
rewards = []
done = False
current_timestep = 0
mppi = MPPIController(num_samples=100, horizon=4, time_offset=0, dynamics_fn=dynamics, cost_fn=cost, data_buffer=data_buffer, lambda_=1.0, sigma=0.2)

start_time = datetime.datetime.now()

while not done:

    a = mppi.control(obs['Zone Air Temperature(SPACE1-1)'], current_timestep)

    obs, reward, done, info = env.step(a)
    rewards.append(reward_(obs))

    # add entry to data buffer using pd.concat
    data_buffer = pd.concat([data_buffer, pd.DataFrame({'zone temperature': obs['Zone Air Temperature(SPACE1-1)'],
                                        'Site Outdoor Air Drybulb Temperature(Environment)': obs['Site Outdoor Air Drybulb Temperature(Environment)'],
                                        'Site Outdoor Air Relative Humidity(Environment)': obs['Site Outdoor Air Relative Humidity(Environment)'],
                                        'Site Wind Speed(Environment)': obs['Site Wind Speed(Environment)'],
                                        'Site Total Solar Radiation Rate per Area(Environment)': obs['Site Diffuse Solar Radiation Rate per Area(Environment)']+obs['Site Direct Solar Radiation Rate per Area(Environment)'],
                                        'Zone People Occupant Count(SPACE1-1)': obs['Zone People Occupant Count(SPACE1-1)'],
                                        'control signal': a}, index=[0])], ignore_index=True)
    
    mppi.update_data_buffer(data_buffer)

    current_timestep += 1

    if current_timestep % 100 == 0:
        checkpoint_time = datetime.datetime.now()
        predicted_total_time = (checkpoint_time - start_time) / current_timestep * 4 * 24 * 92
        predicted_remaining_time = predicted_total_time - (checkpoint_time - start_time)
        predicted_remaining_time_minutes = predicted_remaining_time.seconds / 60
        print('timestep: ', current_timestep, '. time elapsed: ', checkpoint_time - start_time, '. predicted remaining time: ', predicted_remaining_time_minutes, ' minutes')

        # turn reward into a pandas dataframe and save it to a csv file
        reward_df = pd.DataFrame(rewards, columns=['reward'])
        reward_df.to_csv('zimages/reward_plus.csv')
        print('reward saved!')

env.close()

print(
    'Episode ',
    i,
    'Mean reward: ',
    np.mean(rewards),
    'Cumulative reward: ',
    sum(rewards))

# plot the reward, x-axis is the timestep and y-axis is the reward
import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()
plt.savefig('zimages/reward_plus.png')