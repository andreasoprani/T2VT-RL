"""
    Training script, the testing is an exhaustive usage of the fitted regressor
    to speed up.
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
from vectorized_trading_env import VecTradingDerivatives
from trading_env import TradingDerivatives

N_ENVS = 25

def test_policy(env, policy=None):
    """
        Compute a full roll-out of the parallel environment following the policy.
    """
    #rewards, lens, actions = [], [], np.full((env.n_days, env.horizon), np.nan)
    rewards, lengths = np.zeros(env.n_days), np.zeros(env.n_days)
    acts = []
    for day_index in range(0, env.n_days, N_ENVS):
        day_indexes = np.arange(0, env.n_days)[day_index:day_index+N_ENVS]
        ob = env.reset(day_indexes=day_indexes)
        done = [False]
        while not all(done):
            if policy is None:
                actions = [env.action_space.sample() for _ in range(env.n_selected_days)]
            else:
                _, actions = policy.Q.max(ob)
                actions = np.array(actions) + 1 # Add +1 to move from [-1, 1] to [0, 2]
                acts.append(actions[0])
            ob, r, done, _ = env.step(actions)
            rewards[day_indexes] += r
        logging.info("Testing %s"%(day_index,))
    # Scale rewards
    rewards *= 1e5
    return rewards, None

def test_exhaustive(states_actions, fqi_data, policy):
    # Get the values from the fitted regressor
    expected_values = policy.Q.values(states_actions)
    # Construct a synthetic dataset to select the chain
    results = pd.DataFrame.from_dict({
        'day': fqi_data['day'],
        'time_index': fqi_data['time_index'],
        'reward': fqi_data['reward'],
        'action': fqi_data['action'],
        'portfolio': fqi_data['portfolio'],
        'Q': expected_values,
    })
    results = results.set_index(['day', 'time_index', 'portfolio'])
    horizon = fqi_data.time_index.max() + 1
    days = fqi_data.day.unique()
    portfolios = [0] * len(days)
    rewards = np.zeros(len(days))
    for t in range(horizon):
        current_time = results.loc[list(zip(days, [t]*len(days), portfolios))].reset_index()
        max_indexes = current_time.groupby('day').Q.idxmax()
        portfolios = current_time.loc[max_indexes.loc[days]].action.values
        rewards += current_time.loc[max_indexes.loc[days]].reward.values
    return rewards * 1e5

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
        fqi_data = generate(source=train_csv, lag=lag, reward_scale=1)
        logging.info('Generated FQI extended dataset: %s' % (fqi_data.shape, ))
    else:
        fqi_data = pd.read_csv(full_train_data)
        logging.info('Loaded FQI extended dataset: %s' % (fqi_data.shape, ))

    # ======== FQI data preparation ========
    REWARD_SCALE = 1e3
    state_features, next_state_features = feature_selection(strategy='full', lag=lag)
    states_actions = fqi_data[state_features].values
    next_states = fqi_data[next_state_features].values
    rewards = fqi_data['reward'].values * REWARD_SCALE
    absorbing_states = fqi_data['done'].values
    logging.info('Separated columns for FQI.')

    # ======== Testing data prepatation ========
    testing_data = generate(source=test_csv, lag=lag, reward_scale=1)
    REWARD_SCALE = 1e3
    testing_states_actions = testing_data[state_features].values
    testing_next_states = testing_data[next_state_features].values
    testing_rewards = testing_data['reward'].values * REWARD_SCALE
    testing_absorbing_states = testing_data['done'].values
    logging.info('Prepared testing data.')

    # ======== Setting FQI parameters ========
    # Create target environment to test during training
    training_env = VecTradingDerivatives(data=train_csv, n_envs=N_ENVS, maximum_drowdown=-1, time_lag=lag)
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
        rets = test_exhaustive(states_actions, fqi_data, algorithm._policy)
        logging.info('[ITERATION %s] Model fitted: %s'%(i+1, np.sum(rets)))
        testing_rets = test_exhaustive(testing_states_actions, testing_data, algorithm._policy)
        logging.info('[ITERATION %s] Testing: %s'%(i+1, np.sum(testing_rets)))

    # ======== Testing ========
    algorithm._policy.Q.set_regressor_params(n_jobs=1)
    rets, _ = test_policy(training_env, policy=algorithm._policy)
    print("Testing on the training environment:", np.sum(rets))

    if test_csv is not None:
        testing_env = VecTradingDerivatives(data=test_csv, n_envs=N_ENVS, maximum_drowdown=-1, time_lag=lag)
        rets, _ = test_policy(testing_env, policy=algorithm._policy)
        print("Testing on the validation environment:", np.sum(rets))

    # ======== Results ========
    algorithm._policy.Q.set_regressor_params(n_jobs=-1)
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
    parser.add_argument('--iterations', help='Extra-trees min_split', type=int, default=1)
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
