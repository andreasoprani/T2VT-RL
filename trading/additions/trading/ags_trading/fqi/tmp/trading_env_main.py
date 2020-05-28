import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os

class TradingMain(gym.Env):

    metadata = {'render.modes'}

    day_counter = 0

    def __init__(self, fees = 2/100000., spread = 0, maximum_drowdown = -500/100000.):
        #initialize price, profit, portfolio
        self.time_counter = 0
        self.previous_unrealized = 0
        self.unrealized = 0
        self.price = 0
        self.profit = 0
        self.portfolio = 0

        #initialize done, reward and end of the day
        self.done = False
        self.reward = 0
        self.end_of_day = False

        #initialize parameters
        self.fees = fees
        self.spread = spread
        self.maximum_drowdown = maximum_drowdown

        #assign boundaries to observation space for profit, portfolio and time
        self.low_boundaries = np.array([-5, -1, 0])
        self.high_boundaries = np.array([5, 1, 1])

        #define observation space
        self.observation_space = spaces.Box(self.low_boundaries, self.high_boundaries, dtype = np.float32)


    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, target):
        '''target:
        -1: go short
        0: go flat
        1: go long'''
        self.time_counter += 1
        self.actual_target = target
        #Check if the game is done, if is done don't do anything
        if self.done:
            return np.array(self.state), self.reward, self.done, {}

        #REWARD - Take the actual value before the update
        previous_value = self.portfolio * self.price + self.profit
        self.previous_unrealized = self.portfolio * self.price + self.profit

        #Exit game if we reach maximum drowdown
        if self.__is_drawdown(previous_value):  # FIXME: IS DRAWDOWN ############
            return np.array(self.state), self.reward, self.done, {}

        #Update price
        #Set current price to 'price' and update the list episode_prices with
        #with all the element except the first
        self.vector_price = self.episode_prices[:self.vector_length()]
        price, self.episode_prices = self.vector_price[-1], self.episode_prices[1:]

        #Check if we are at the end of the day
        #In this case set action to 0 (we have to finish flat) and set status to done.
        if len(self.episode_prices) + 1 == len(self.vector_price):
            #price = self.episode_prices[-1]
            self.done = True
            self._set_price(price)
            self.end_of_day = True

            self.__go_flat()
            self.__update_reward(previous_value)

            return np.array(self.state), self.reward, self.done, {}


        #Update price and state with the new price vector
        self._set_price(price)
        self.state = self.state_vector()

        #GO SHORT (sell -> -1)
        if target == -1:
            self.__go_short()

        #GO FLAT (bring portfolio to 0)
        if target == 0:
            self.__go_flat()

        #GO LONG (buy -> +1)
        if target == 1:
            self.__go_long()


        self.unrealized = self.portfolio * self.price + self.profit
        self.state = self.state_vector()
        self.__update_reward(previous_value)

        return np.array(self.state), self.reward, self.done, {}


    def __update_reward(self, previous_value):
        #ASSIGN REWARD - reward is assigned after the update
        actual_value = self.portfolio * self.price + self.profit

        #TEST: moltiplicare il reward per scalarlo a valori più vicini ai prezzi
        self.reward = (actual_value - previous_value) # FIXME: lo togliamo per testare* 5e3


    #GO FLAT FUNCTION
    def __go_flat(self):
        #GO FLAT (bring portfolio to 0)
        #buy to go back to 0
        if self.portfolio == -1:
            self.portfolio = 0
            self.profit = self.profit - self.price - self.fees
            self.state = self.state_vector()

        #sell
        elif self.portfolio == +1:
            self.portfolio = 0
            self.profit = self.profit + self.price - self.fees
            self.state = self.state_vector()

        else:
            self.state = self.state_vector()

    #GO SHORT FUNCTION
    def __go_short(self):
        #GO SHORT (sell -> -1)
        #single sell
        if self.portfolio == 0:
            self.portfolio = -1
            self.profit = self.profit + self.price - self.fees
            self.state = self.state_vector()

        #double sell
        elif self.portfolio == 1:
            self.portfolio = -1
            self.profit = self.profit + self.price - self.fees
            self.profit = self.profit + self.price - self.fees
            self.state = self.state_vector()

        else:
            self.state = self.state_vector()


    #GO LONG FUNCTION
    def __go_long(self):
        #GO LONG (buy -> +1)
        #single buy
        if self.portfolio == 0:
            self.portfolio = 1
            self.profit = self.profit - self.price - self.fees
            self.state = self.state_vector()

        #double buy
        elif self.portfolio == -1:
            self.portfolio = 1
            self.profit = self.profit - self.price - self.fees
            self.profit = self.profit - self.price - self.fees
            self.state = self.state_vector()

        else:
            self.state = self.state_vector()



    def _set_price(self, price):
        '''set the actual price'''
        self.price = price

    def render(self, arg):
        pass


    def __is_drawdown(self, previous_value):
        '''Check if we reach drawdown. In this case set status to done, go flat and assign regret to
        the reward'''
        if previous_value < self.maximum_drowdown:
            price = self.episode_prices[self.vector_length()-1]
            self._set_price(price)
            #print('price', price)

            self.__go_flat()
            self.__update_reward(previous_value)
            #self.reward -= 100 #FIXME:
            self.__regret_final(previous_value)

            self.done = True

            self.vector_price = self.episode_prices[:self.vector_length()]
            price, self.episode_prices = self.vector_price[-1], self.episode_prices[1:]

            self.state = self.state_vector()
            #LOG
            #self.write_log()

            return True

        else:
            return False


    def __regret_final(self, previous_value):
        #regret is multiplied by a factor n to prevent early suicide of the agent
        return abs(self.price - self.episode_prices[-1]) * 10000

    def vector_length(self):
        #used in child class
        pass


    def state_vector(self):
        #used in child class
        pass


    def day_percentage(self, episode_prices):
        return 1 - (len(episode_prices) + 1 - self.vector_length()) / self.day_len


    '''
    def day_percentage(self):
        sin_1 = [ np.sin(1 * (self.time_counter/1200)),
                  np.sin(2 * (self.time_counter/1200)),
                  np.sin(3 * (self.time_counter/1200)),
                  np.sin(5 * (self.time_counter/1200)),
                  np.sin(8 * (self.time_counter/1200))
                ]

        day_percentage = [1 - (len(self.episode_prices) + 1 - self.vector_length()) / self.day_len]

        return np.array(sin_1 + day_percentage)
    '''

    def reset(self):
        self.time_counter = 0
        self.previous_unrealized = 0
        self.unrealized = 0
        self.change_state = 0
        self.hold_state = 0
        self.profit = 0
        self.portfolio = 0
        self.done = False
        self.end_of_day = False

        if True: #np.random.randint(10000) < 100:
            if TradingMain.day_counter < len(TradingMain.days) - 1:
                TradingMain.day_counter += 1
            else:
                TradingMain.day_counter = 0

            self.day = TradingMain.days[TradingMain.day_counter]
            #print('Day changed, loaded {}'.format(self.day))
            TradingMain.last_day_loaded = self.day

        return self.state
