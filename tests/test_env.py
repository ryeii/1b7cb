import os
from random import randint, sample

import gym

from sinergym.utils.constants import *
from sinergym.utils.env_checker import check_env


def test_reset(env_demo):
    obs = env_demo.reset()
    assert len(obs) == len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + \
        4  # year, month, day and hour
    assert env_demo.simulator._episode_existed


def test_step(env_demo):
    env_demo.reset()
    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + \
        4  # year, month, day and hour
    assert not isinstance(reward, type(None))
    assert not done
    assert list(
        info.keys()) == [
        'timestep',
        'time_elapsed',
        'year',
        'month',
        'day',
        'hour',
        'total_power',
        'total_power_no_units',
        'comfort_penalty',
        'abs_comfort',
        'temperatures',
        'out_temperature',
        'action_']
    assert info['timestep'] == 1
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']

    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 20
    assert not isinstance(reward, type(None))
    assert not done
    assert list(
        info.keys()) == [
        'timestep',
        'time_elapsed',
        'year',
        'month',
        'day',
        'hour',
        'total_power',
        'total_power_no_units',
        'comfort_penalty',
        'abs_comfort',
        'temperatures',
        'out_temperature',
        'action_']
    assert info['timestep'] == 2
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']


def test_close(env_demo):
    env_demo.reset()
    assert env_demo.simulator._episode_existed
    env_demo.close()
    assert not env_demo.simulator._episode_existed
    assert env_demo.simulator._conn is None


def test_all_environments():

    envs_id = [env_spec.id for env_spec in gym.envs.registry.all()
               if env_spec.id.startswith('Eplus')]
    # Select 10 environments randomly (test would be too large)
    samples_id = sample(envs_id, 5)
    for env_id in samples_id:
        # Create env with TEST name
        env = gym.make(env_id)

        # stable_baselines 3 environment checker. Check if environment follows
        # Gym API.
        check_env(env)

        # Rename directory with name TEST for future remove
        os.rename(env.simulator._env_working_dir_parent, 'Eplus-env-TEST' +
                  env.simulator._env_working_dir_parent.split('/')[-1])

        env.close()
