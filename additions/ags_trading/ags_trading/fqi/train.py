"""
    Training script.

    Notes:
    - The last state_action feature must be the action
"""

# Library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, pickle, argparse, logging
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.utilities.evaluation import evaluate_policy
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from ags_trading.trading_env import TradingDerivatives

def test_policy(env, policy=None):
    """
        Compute a full roll-out of the environment following the policy
    """
    rewards, lens, actions = [], [], np.full((env.n_days, env.horizon), np.nan)
    for day_index in range(env.n_days):
        ob = env.reset(day_index=day_index)
        done, day_reward, t = False, 0.0, 0
        while not done:
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy.sample_action(ob)[0] + 1 # Add +1 to go from [-1, 1] to [0, 2]
            ob, r, done, _ = env.step(action)
            day_reward += r * 1e5
            actions[day_index, t] = action
            t += 1
        rewards.append(day_reward)
        lens.append(t)
        print('done day', day_reward)
    return rewards, actions

def feature_selection(strategy='full', lag=60):
    """
        Returns a list of all the features selected for each named strategy,
        in the form: (state_action_features, next_state_features, reward_features).
    """
    if strategy == 'full':
        return (['D'+str(i) for i in range(0, lag)]+['portfolio', 'time', 'action'], # state_action
                ['D'+str(i-1) for i in range(0, lag)]+['action', 'time'], # next_state (action is next_portfolio)
                )
    else:
        raise Exception('Unrecognized feature selection strategy.')

def train(train_csv=None, test_csv=None, full_train_data=None, min_split=1, lag=60,
            iterations=10, batch_size=10):
    # Input checking
    assert (full_train_data is not None) or (train_csv is not None), 'No expanded training dataset provided for FQI.'
    logging.info('Starting...')

    # ======== FQI data loading ========
    if full_train_data is None:
        # Generate dataset directly
        from dataset_generation import generate
        fqi_data = generate(source=train_csv, lag=lag)
        logging.info('Generated FQI extended dataset: %s' % (fqi_data.shape, ))
    else:
        fqi_data = pd.read_csv(full_train_data)
        logging.info('Loaded FQI extended dataset: %s' % (fqi_data.shape, ))

    # ======== FQI data preparation ========
    state_features, next_state_features = feature_selection(strategy='full', lag=lag)
    states_actions = fqi_data[state_features].values
    next_states = fqi_data[next_state_features].values
    rewards = fqi_data['reward'].values
    absorbing_states = fqi_data['done'].values
    logging.info('Separated columns for FQI.')

    # ======== Setting FQI parameters ========
    # Create target environment to test during training
    training_env = TradingDerivatives(data=train_csv, maximum_drowdown=-1)
    logging.info('Creating training environment.')
    regressor_params = {'n_estimators': 50,
                        'criterion': 'mse',
                        'min_samples_split': min_split,
                        'min_samples_leaf': 1,
                        'n_jobs': -1}
    actions = [-1, 0, 1]
    pi = EpsilonGreedy(actions, ZeroQ(), epsilon=0) # Greedy policy
    # Baseline score for the environment
    rets, _ = test_policy(training_env)
    logging.info('Random policy total profit: %s'%(np.sum(rets), ))

    # Create algorithm
    algorithm = FQI(training_env, pi, verbose = False, actions = actions,
                    batch_size = batch_size, max_iterations = iterations,
                    regressor_type = ExtraTreesRegressor, **regressor_params)
    logging.info('Algorithm set up, ready to go.')

    # ======== Training Loop ========
    for i in range(iterations):
        algorithm._iter(states_actions, rewards, next_states, absorbing_states)
        logging.info('[ITERATION %s] Metric:'%(i+1,))
        pi.Q.set_regressor_params(n_jobs=1)
        rets, _ = test_policy(training_env, policy=algorithm._policy)
        pi.Q.set_regressor_params(n_jobs=-1)
        logging.info('[ITERATION %s] Testing: %s'%(i+1, np.sum(rets)))

    # ======== Testing ========


    # ======== Results ========
    logging.info('End.')

if __name__ == '__main__':
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data sources
    parser.add_argument('--train_csv', help='CSV containing training prices for the environment.', type=str, default=None)
    parser.add_argument('--test_csv', help='CSV containing test prices for the environment. If None, no test is performed.', type=str, default=None)
    parser.add_argument('--full_train_data', help='CSV containing the expanded training data.', type=str, default=None)
    # Parameters
    parser.add_argument('--min_split', help='Extra-trees min_split', type=int, default=1)
    parser.add_argument('--log', help='Log level', default='INFO')
    parser.add_argument('--lag', help='Number of previous prices to include.', type=int, default=60)
    args = parser.parse_args()
    # Logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    assert isinstance(numeric_level, int), 'Invalid log level: %s' % loglevel
    logging.basicConfig(level=numeric_level, format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    args = vars(args)
    del args['log']
    # Call the train function with arguments
    train(**args)
