import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os
from trading_env_main import TradingMain

class Trading60Discrete(TradingMain):

    def __init__(self, csv_path):
        self.__data_length = 60 #(9)
        super().__init__()

        TradingMain.df = pd.read_csv(csv_path, index_col = 0) #FIXME:
        TradingMain.days = TradingMain.df['day'].unique() #FIXME:
        TradingMain.last_day_loaded = TradingMain.days[TradingMain.day_counter] #FIXME:

        self.action_space = spaces.Discrete(3)
        self.action_space.n = 3

        # FIXME: aggiungere valori per FQI
        self.action_dim = 1 ## FQI ##
        self.state_dim = 61 # aggiunto per FQI (10)  ##################
        self.gamma = 1 # aggiunto per FQI ############
        self.horizon = 1167 # ##################################

        #Initialize array of 60 values
        min_obs = np.zeros(self.__data_length)
        max_obs = np.ones(self.__data_length) * 2 #set the max EUR-USD value to 2

        #create the observation space
        #boundaries refers to price, profit and portfolio
        min_obs = np.append(min_obs, self.low_boundaries)
        max_obs = np.append(max_obs, self.high_boundaries)

        self.low_boundaries = min_obs
        self.high_boundaries = max_obs

        self.observation_space = spaces.Box(min_obs, max_obs, dtype = np.float32)

        self.state = self.__feed_data()

        #log initial state
        #self.write_log() #FIXME: write log !!!


    def step(self, target):
        '''
        if target > 1.333:
            target = 1
        elif target < 0.666:
            target = -1
        else:
            target = 0
        '''

        return super().step(target)


    def vector_length(self):
        return self.__data_length

    #return np.append(np.append(np.append(self.vector_price, self.portfolio * self.price + self.profit),
    #                                    self.portfolio), self.day_percentage(self.episode_prices))

    #        return np.append(np.append(self.portfolio, self.day_percentage(self.episode_prices)), # FIXME: modifico per FQI
#                         self.vector_price)
    def state_vector(self):
        return np.append(self.portfolio, self.vector_price)  # FIXME: modifico per FQI

    def __feed_data(self):
        self.episode_prices = TradingMain.df[TradingMain.df['day'] == TradingMain.last_day_loaded]['open'].values
        self.day_len = len(self.episode_prices)

        #At the reset set the price to the first price after 60 minutes
        #State will be the price for the previous 60 minutes
        self.vector_price = self.episode_prices[:self.__data_length]
        #Set the first element of the list of prices as the actual price
        price, self.episode_prices = self.vector_price[-1], self.episode_prices[1:]
        self._set_price(price)

        self.state = self.state_vector()

        return self.state

    def reset(self, state=None):
        super().reset()
        return self.__feed_data()
