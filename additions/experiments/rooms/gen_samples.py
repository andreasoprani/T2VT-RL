import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../.."))

import numpy as np
from envs.two_room_gw import TwoRoomGridworld
from envs.three_room_gw import ThreeRoomGridworld
from features.agrbf import build_features_gw_state
from additions.approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from algorithms.nt import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime

# Global parameters
render = False
verbose = False

seed = 1
np.random.seed(seed)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=50)
parser.add_argument("--max_iter", default=30000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.2)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.02)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=50)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--env", default="two-room-gw")
parser.add_argument("--gw_size", default=10)
parser.add_argument("--n_basis", default=11)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--timesteps", default=10)
parser.add_argument("--samples_per_timestep", default=5)
parser.add_argument("--doors_std", default=0.2)
parser.add_argument("--sources_file_name", default=path + "/sources-2r")
parser.add_argument("--tasks_file_name", default=path + "/tasks-2r")

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
max_iter = int(args.max_iter)
buffer_size = int(args.buffer_size)
random_episodes = int(args.random_episodes)
exploration_fraction = float(args.exploration_fraction)
eps_start = float(args.eps_start)
eps_end = float(args.eps_end)
train_freq = int(args.train_freq)
eval_freq = int(args.eval_freq)
mean_episodes = int(args.mean_episodes)
alpha = float(args.alpha)
env = str(args.env)
gw_size = int(args.gw_size)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
timesteps = int(args.timesteps)
samples_per_timestep = int(args.samples_per_timestep)
doors_std = float(args.doors_std)
sources_file_name = str(args.sources_file_name)
tasks_file_name = str(args.tasks_file_name)

# Seed to get reproducible results
seed = 1
np.random.seed(seed)

# Generate tasks

# door 1 ----->
# door 2 <-----
# the standard deviation for doors is added to the boundaries to prevent too much clipping
doors_means = np.linspace(0.5 + doors_std, gw_size - 0.5 - doors_std, timesteps+1)
doors2_means = np.flip(doors_means)

mdps = []

for t in range(timesteps + 1):
    # Generate random door positions
    doors = np.random.normal(loc = doors_means[t], scale = doors_std, size = samples_per_timestep)
    doors2 = np.random.normal(loc = doors2_means[t], scale = doors_std, size = samples_per_timestep)
    # Clip
    doors = [0.5 if x < 0.5 else (gw_size - 0.5 if x > gw_size - 0.5 else x) for x in doors]
    doors2 = [0.5 if x < 0.5 else (gw_size - 0.5 if x > gw_size - 0.5 else x) for x in doors2]
    # Append list of mdps
    if env == "two-room-gw":
        mdps.append([TwoRoomGridworld(np.array([gw_size, gw_size]), door_x=d) for d in doors])
    elif env == "three-room-gw":
        mdps = [ThreeRoomGridworld(np.array([gw_size, gw_size]), door_x=(d1,d2)) for (d1,d2) in zip(doors,doors2)]

eval_states = [np.array([0., 0.]) for _ in range(10)]

state_dim = mdps[0][0].state_dim
action_dim = 1
n_actions = mdps[0][0].action_space.n
K = n_basis ** 2

# Create BellmanOperator
operator = MellowBellmanOperator(kappa, tau, xi, mdps[0][0].gamma, K, action_dim)
# Create Q Function
Q = MLPQFunction(K, n_actions, layers=None)
# Create RBFs
rbf = build_features_gw_state(gw_size, n_basis, state_dim)


def run(mdp, seed=None):
    return learn(mdp,
                 Q,
                 operator,
                 max_iter=max_iter,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 alpha=alpha,
                 train_freq=train_freq,
                 eval_freq=eval_freq,
                 eps_start=eps_start,
                 eps_end=eps_end,
                 exploration_fraction=exploration_fraction,
                 random_episodes=random_episodes,
                 eval_states=eval_states,
                 mean_episodes=mean_episodes,
                 preprocess=rbf,
                 seed=seed,
                 render=render,
                 verbose=verbose)

# Learn optimal policies for all sources
results = []
for i in range(timesteps):
    print("Timestep", i)
    results.append([run(mdps[i][j], seed) for j in range(samples_per_timestep)])
    print("Door positions:", [t[0][1] for t in results[-1]])
    print("Last rewards:", [t[2][3][-1] for t in results[-1]], "\n")

# Save sources to file
utils.save_object(results, sources_file_name)

# Save tasks to file
tasks = mdps[-1]
print("Tasks")
print("Door positions:", [t.get_info()[1] for t in tasks])
utils.save_object(tasks, tasks_file_name)