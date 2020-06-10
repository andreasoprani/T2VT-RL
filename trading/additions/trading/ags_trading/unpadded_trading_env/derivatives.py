'''
    Subclass of trading main environment, observations are derivatives of prices
    in the previous N minutes.
'''

from ags_trading.unpadded_trading_env import TradingMain
from gym import spaces
import numpy as np

MAX_DERIVATIVE = 5.0

class TradingDerivatives(TradingMain):

    def __init__(self, time_lag=60, **kwargs):
        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        observation_low = np.concatenate([np.full((time_lag), -MAX_DERIVATIVE), [-1.0, 0.0]])
        observation_high = np.concatenate([np.full((time_lag), +MAX_DERIVATIVE), [+1.0, +1.0]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        # Internals
        self.derivatives = None
        self.time_lag = time_lag
        # Required for FQI
        self.action_dim = 1
        self.state_dim = self.observation_space.shape[0]
        self.gamma = 0.9999

    def _normalize(self, X):
        return (X)/self.std_delta
        # return (X-self.mean_delta)/self.std_delta

    def _denormalize(self, X):
        return X*self.std_delta#+self.mean_delta

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        lagged_derivatives = self.derivatives[self.current_timestep+1:self.current_timestep+self.time_lag+1][::-1]
        return np.concatenate([lagged_derivatives, [self.current_portfolio, self.current_timestep / self.horizon]])

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        return (self.current_portfolio * (self.current_price - self.previous_price) - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees) #percentage

    def reset(self, initial_state=None):
        # initial_state is present only to respect the format of the other environments
        super().reset()
        # Compute derivatives once for all, pad with time_lag zeros
        self.derivatives = self._normalize(np.insert(self.prices[1:] - self.prices[:-1],0,0))
        return self._observation()

    def get_info(self):
        return ["TradingDerivatives", self.csv_path]

# UNIT TESTING
if __name__ == '__main__':
    import pandas as pd
    sample_dataset = pd.DataFrame.from_dict({
        'open': [0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3],
        'day': [1, 1, 1, 1, 2, 2, 2, 2],
        'count': [1, 1, 1, 1, 3, 3, 3, 3],
    })
    env = TradingPrices(data=sample_dataset)
    ob, done, reward, t = env.reset(), False, 0.0, 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        reward += rew
        t += 1
    print('Final reward:', reward)
    print('Final length:', t)



class TradingDerivativesWithStateReward(TradingDerivatives):

    def __init__(self, time_lag=60, **kwargs):
        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        observation_low = np.concatenate([np.full((time_lag), -MAX_DERIVATIVE), [-1.0, 0.0]]*2)
        observation_high = np.concatenate([np.full((time_lag), +MAX_DERIVATIVE), [+1.0, +1.0]]*2)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.last_reward=0

    def reset(self):
        x = super().reset()
        self.last_x = x
        return np.concatenate((self.last_x, x))

    def step(self, action):
        step_observation, step_reward, step_done, _ = super().step(action)
        last_x = self.last_x
        reward = self.last_reward
        self.last_reward = step_reward
        self.last_x = step_observation
        obs = np.concatenate((last_x, step_observation))
        return obs, reward,  step_done, {}
    
    def get_info(self):
        return ["TradingDerivativesWithStateReward", self.csv_path]