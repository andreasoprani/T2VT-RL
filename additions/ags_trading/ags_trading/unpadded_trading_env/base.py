import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os
#import custompaths

class TradingMain(gym.Env):
    """
        Abstract class which implements the trading actions. Must be extended for
        different types of observations and rewards.
    """

    metadata = {'render.modes'}
# fees = 2/100000.
    def __init__(self, csv_path=None, fees = 2/100000, spread = 0, maximum_drowdown = -500/100000., norm_prices=False, time_lag=60, old_csv_path=None, testing=False):
        # Check data (prices CSV)
        assert csv_path is not None, "Need price data to create environment."
        csv_path = os.path.join(os.path.expanduser('~') + '/NSD-RL+/ags_trading/env_settings', csv_path)
        if old_csv_path is None:
            old_csv_path = csv_path
        else:
            csv_path = os.path.join(os.path.expanduser('~') + '/NSD-RL/trading_env/ags_trading/env_settings', csv_path)
        self.norm_prices = norm_prices
        self.time_lag = time_lag
        self.data = pd.read_csv(csv_path, index_col = 0) #FIXME:
        self.days = self.data['day'].unique()
        
        # mean and standard devs for the training dataset
        old_data = pd.read_csv(old_csv_path, index_col = 0)
        train_prices = self.data['open'].values
        self.mean_price = np.mean(train_prices)
        self.std_dev_prices = np.sqrt(np.var(train_prices))
        deltas=(train_prices[1:]-train_prices[:-1])#/prices[:-1]
        self.mean_delta = np.mean(deltas)
        self.std_delta = np.sqrt(np.var(deltas))

        props = (train_prices[1:] - train_prices[:-1]) / train_prices[:-1]
        self.mean_props = np.mean(props)
        self.std_props = np.sqrt(np.var(props))

        self.n_days = len(self.days)
        self.horizon = self.data.shape[0] // self.n_days
        # Initialize parameters
        self.fees = fees
        self.spread = spread
        self.maximum_drowdown = maximum_drowdown
        # Internal variables
        self.done = True
        self.prices = None
        self.current_timestep = 0
        # Prices
        self.current_price = None
        self.previous_price = None
        # Portfolio
        self.current_portfolio = None
        self.previous_portfolio = None
        if testing:
            self.starting_day_index = 0
        else:
            self.starting_day_index = None

    def seed(self, seed = None):
        np.random.seed(seed)
        random.seed(seed)

    def _observation(self):
        raise Exception('Not implemented in abstract class.')

    def _reward(self):
        raise Exception('Not implemented in abstract class.')

    def step(self, action):
        """
            Act on the environment. Target can be either a float from -1 to +1
            of an integer in {-1, 0, 1}. Float represent partial investments.
        """
        # Check if the environment has terminated.
        if self.done:
            return self._observation(), 0.0, True, {} # FIXME Should I stay here?

        # Transpose action if action space is discrete [0, 2] => [-1, +1]
        if isinstance(self.action_space, spaces.Discrete):
            action = action - 1

        # Check the action is in the range
        assert -1 <= action <= +1, "Action not in range!"

        # Update price
        self.current_timestep += 1
        self.previous_price, self.current_price = self.current_price, self.prices[self.current_timestep+self.time_lag]

        # Check if day has ended
        self.done = self.current_timestep >= (len(self.prices) - self.time_lag - 1)

        # Perform action
        self.previous_portfolio, self.current_portfolio = self.current_portfolio, action

        # Compute the reward and update the profit
        reward = self._reward()
        self.profit += reward

        # Check if drawdown condition is met
        self.done = self.done #or self.profit < self.maximum_drowdown

        return self._observation(), reward,  self.done, {}

    def reset(self):
        # Extract day from data and set prices
        if self.starting_day_index is None:
            selected_day = np.random.choice(self.days)
        else:
            selected_day = self.days[self.starting_day_index]
            self.starting_day_index = (self.starting_day_index + 1) % len(self.days)
        self.selected_day = selected_day
        self.selected_data = self.data[self.data['day'] == selected_day]
        if self.norm_prices:
            self.prices = (self.selected_data['open'].values-self.mean_price)/self.std_dev_prices
        else:
            self.prices = self.selected_data['open'].values
        # Init internals
        self.current_timestep = 0
        self.current_price = self.prices[self.current_timestep + self.time_lag]
        self.previous_price = None
        self.current_portfolio = 0
        self.previous_portfolio = None
        self.done = False
        self.profit = 0
        return None
