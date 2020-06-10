import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

import gym
from ags_trading.unpadded_trading_env.derivatives import TradingDerivatives

import numpy as np
from additions.approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from additions.algorithms.mgvt_torch import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
import glob
import errno

# Global parameters
verbose = False

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=10000)
parser.add_argument("--max_iter", default=100000000)
parser.add_argument("--buffer_size", default=1000000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--train_freq", default=1000)
parser.add_argument("--eval_freq", default=100000)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha_adam", default=0.001)
parser.add_argument("--alpha_sgd", default=0.0001)
parser.add_argument("--lambda_", default=0.0001)
parser.add_argument("--time_coherent", default=False)
parser.add_argument("--n_weights", default=10)
parser.add_argument("--cholesky_clip", default=0.0001)
parser.add_argument("--l1", default=32)
parser.add_argument("--l2", default=32)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=20)
parser.add_argument("--eta", default=1e-6)  # learning rate for
parser.add_argument("--eps", default=0.001)  # precision for the initial posterior approximation and upperbound tighting
parser.add_argument("--bandwidth", default=.00001)  # Bandwidth for the Kernel Estimator
parser.add_argument("--post_components", default=1)  # number of components of the posterior family
parser.add_argument("--max_iter_ukl", default=60)

parser.add_argument("--source_file", default=path + "/sources")
parser.add_argument("--tasks_file", default=path + "/tasks")
parser.add_argument("--load_results", default = False)

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
max_iter = int(args.max_iter)
buffer_size = int(args.buffer_size)
random_episodes = int(args.random_episodes)
train_freq = int(args.train_freq)
eval_freq = int(args.eval_freq)
mean_episodes = int(args.mean_episodes)
alpha_adam = float(args.alpha_adam)
alpha_sgd = float(args.alpha_sgd)
lambda_ = float(args.lambda_)
time_coherent = bool(int(args.time_coherent))
n_weights = int(args.n_weights)
cholesky_clip = float(args.cholesky_clip)
l1 = int(args.l1)
l2 = int(args.l2)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
eps = float(args.eps)
eta = float(args.eta)
post_components = int(args.post_components)
bandwidth = float(args.bandwidth)
max_iter_ukl = int(args.max_iter_ukl)

source_file = str(args.source_file)
tasks_file = str(args.tasks_file)
load_results = bool(args.load_results)

# Seed to get reproducible results
seed = 1
np.random.seed(seed)

file_path = "results/trading/"
if not os.path.exists(file_path):
    os.mkdir(file_path)

if load_results:
    file_name = "mgvt_" + str(post_components) + "c_"
    f = file_path + file_name + "*.pkl"
    fs = glob.glob(f)
    if len(fs) == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)
    file_name = fs[0][:-4]
else:
    file_name = file_path + "mgvt_" + str(post_components) + "c_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load tasks
mdps = utils.load_object(tasks_file)

n_eval_episodes = 5

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n

# Create BellmanOperator
operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)
# Create Q Function
layers = [l1]
if l2 > 0:
    layers.append(l2)
Q = MLPQFunction(state_dim, n_actions, layers=layers)

def run(mdp, seed=None):
    return learn(mdp,
                 Q,
                 operator,
                 max_iter=max_iter,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 alpha_adam=alpha_adam,
                 alpha_sgd=alpha_sgd,
                 lambda_=lambda_,
                 n_weights=n_weights,
                 train_freq=train_freq,
                 eval_freq=eval_freq,
                 random_episodes=random_episodes,
                 eval_episodes=n_eval_episodes,
                 mean_episodes=mean_episodes,
                 cholesky_clip=cholesky_clip,
                 bandwidth=bandwidth,
                 post_components=post_components,
                 max_iter_ukl=max_iter_ukl,
                 eps=eps,
                 eta=eta,
                 time_coherent=time_coherent,
                 source_file=source_file,
                 seed=seed,
                 verbose=verbose,
                 ukl_tight_freq=1)

seeds = [9, 44, 404, 240, 259, 141, 371, 794, 41, 507, 819, 959, 829, 558, 638, 127, 672, 4, 635, 687]
seeds = seeds[:n_runs]

mdps = np.random.choice(mdps, len(seeds))

if load_results:
    old_results = utils.load_object(file_name)
    skip = len(old_results)
    mdps = mdps[skip:]
    seeds = seeds[skip:]

if n_jobs == 1:
    results = [run(mdp,seed) for (mdp,seed) in zip(mdps,seeds)]
elif n_jobs > 1:
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

if load_results:
    old_results.extend(results)
    utils.save_object(old_results, file_name)
else:
    utils.save_object(results, file_name)