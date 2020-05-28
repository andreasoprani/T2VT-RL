import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os

class VecTradingMain(gym.Env):
    """
        Abstract class which implements the trading actions. Must be extended for
        different types of observations and rewards. Extends the vec environment
    """

    metadata = {'render.modes'}

    def __init__(self, csv_path=None, n_envs=1, fees = 2/100000., spread = 0, maximum_drowdown = -500/100000.):
        # Check data (prices CSV)
        self.csv_path = csv_path

        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../env_settings"
        assert csv_path is not None, "Need price data to create environment."
        csv_path = os.path.join(dir_path, csv_path)
        old_csv_path = csv_path
        
        self.data = pd.read_csv(csv_path, index_col = 0) #FIXME:
        self.days = self.data['day'].unique()
        self.n_days = len(self.days)
        self.horizon = self.data.shape[0] // self.n_days
        # Initialize parameters
        self.fees = fees
        self.spread = spread
        self.maximum_drowdown = maximum_drowdown
        # Internal variables
        self.n_envs = n_envs
        self.dones = [True] * self.n_envs
        self.prices = None
        self.current_timestep = 0
        # Prices
        self.current_prices = None
        self.previous_prices = None
        # Portfolio
        self.current_portfolios = None
        self.previous_portfolios = None

    def seed(self, seed = None):
        np.random.seed(seed)
        random.seed(seed)

    def _observation(self):
        raise Exception('Not implemented in abstract class.')

    def _reward(self):
        raise Exception('Not implemented in abstract class.')

    def step(self, actions):
        """
            Act on the environment. Target can be either a float from -1 to +1
            of an integer in {-1, 0, 1}. Float represent partial investments.
        """
        # Check if the environment has terminated.
        if all(self.dones):
            return self._observation, [0.0]*self.n_selected_days, self.dones, {}

        # Transpose action if action space is discrete [0, 2] => [-1, +1]
        if isinstance(self.action_space, spaces.Discrete):
            actions = np.array(actions) - 1

        # Check actions are in range [-1, +1]
        assert all([-1 <= a <= 1 for a in actions]), "Actions not in range!"

        # If only some environments are ended, set their actions to zero
        actions = np.array([a if not done else 0 for (a, done) in zip(actions, self.dones)])

        # Update price
        self.current_timestep += 1
        self.previous_prices, self.current_prices = self.current_prices, self.prices[:,self.current_timestep]

        # Check if day has ended
        if self.current_timestep >= (self.prices.shape[1] - 1):
            self.dones = [True] * self.n_selected_days

        # Perform action
        self.previous_portfolios, self.current_portfolios = self.current_portfolios, actions

        # Compute the reward and update the profit
        rewards = self._reward()
        rewards = [r if not done else 0 for (r, done) in zip(rewards, self.dones)]
        self.profits += rewards

        # Check if drawdown condition is met
        self.dones = [done or profit < self.maximum_drowdown for (done, profit) in zip(self.dones, self.profits)]

        return self._observation(), rewards, self.dones, {}

    def reset(self, day_indexes=None):
        # Extract day from data and set prices
        if day_indexes is None:
            self.selected_days = np.random.choice(self.days, size=self.n_envs, replace=False)
        else:
            self.selected_days = self.days[day_indexes]
        self.n_selected_days = len(self.selected_days)
        self.selected_data = self.data[self.data['day'].isin(self.selected_days)]
        self.prices = np.reshape(self.selected_data['open'].values, (self.n_selected_days, self.horizon))
        # Init internals
        self.current_timestep = 0
        self.current_prices = self.prices[:,self.current_timestep]
        self.previous_prices = None
        self.current_portfolios = np.zeros(self.n_selected_days)
        self.previous_portfolios = None
        self.dones = [False] * self.n_selected_days
        self.profits = np.zeros(self.n_selected_days)
        return None
