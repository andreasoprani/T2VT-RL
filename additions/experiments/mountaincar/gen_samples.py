import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../.."))

import numpy as np
from envs.mountain_car import MountainCarEnv
from additions.approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from additions.algorithms.nt import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.dqn import DQN

# Global parameters
render = False
verbose = False

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_iter", default=1500000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.5)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.01)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=64)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha", default=0.001)
# Car parameters (default = randomize)
parser.add_argument("--speed", default=-1)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--dqn", default=False)

parser.add_argument("--timesteps", default=10)
parser.add_argument("--samples_per_timestep", default=5)
parser.add_argument("--min_speed", default=0.001)
parser.add_argument("--max_speed", default=0.0015)
parser.add_argument("--speed_std", default=0.00002)
parser.add_argument("--just_one_timestep", default=-1) # Used to re-train for just one timestep. -1 = False, 0 -> (timesteps - 1) = timestep to re-train 
parser.add_argument("--experiment_type", default="linear")
parser.add_argument("--sources_file_name", default=path + "/sources")
parser.add_argument("--tasks_file_name", default=path + "/tasks")

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
l1 = int(args.l1)
l2 = int(args.l2)
alpha = float(args.alpha)
speed = float(args.speed)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
dqn = bool(args.dqn)

timesteps = int(args.timesteps)
samples_per_timestep = int(args.samples_per_timestep)
min_speed = float(args.min_speed)
max_speed = float(args.max_speed)
speed_std = float(args.speed_std)
just_one_timestep = int(args.just_one_timestep)
experiment_type = str(args.experiment_type)
sources_file_name = str(args.sources_file_name)
tasks_file_name = str(args.tasks_file_name)

sources_file_name += "-" + experiment_type
tasks_file_name += "-" + experiment_type

# Seed to get reproducible results
seed = 1
np.random.seed(seed)

def gen_speed_means(exp_type="linear"):

    if exp_type == "sin":
        # sin(x) normalized on [min, max], x = 2pi * (i/(t+1))
        s_means = np.sin((2 * np.pi) * np.linspace(0, 1, timesteps + 1))
        s_means = s_means * ((max_speed - min_speed) / 2) + (max_speed + min_speed) / 2
        
        return s_means

    if exp_type == "periodic-no-rep":
        # as sin but x in [a, pi + a] where a is in [pi/4, pi/2]
        a = np.random.uniform(low = np.pi/4, high = np.pi/2)
        s_means = np.sin(np.linspace(a, a + np.pi, timesteps + 1))
        s_means = s_means * ((max_speed - min_speed) / 2) + (max_speed + min_speed) / 2
        
        return s_means

    if exp_type == "polynomial":
        # polynomial of fourth order fit on the points (0,-1), (0.2,0), (0.5,0), (0.7,0), (1,1)
        a = -15.625 # x^4
        b = 39.5833 # x^3
        c = -31.875 # x^2 
        d = 9.91667 # x
        e = -1

        f = lambda x : a * x**4 + b * x**3 + c * x**2 + d * x + e
        s_means = np.linspace(0, 1, timesteps + 1)
        s_means = f(s_means)
        s_means = s_means * ((max_speed - min_speed) / 2) + (max_speed + min_speed) / 2
        
        return s_means

    else:
        # min -> max
        s_means = np.linspace(min_speed, max_speed, timesteps + 1)
        return s_means
    
speed_means = gen_speed_means(experiment_type)

mdps = []

for t in range(timesteps + 1):
    # Generate random speed
    speeds = np.random.normal(loc = speed_means[t], scale = speed_std, size = samples_per_timestep)
    print(speeds)
    mdps.append([MountainCarEnv(s) for s in speeds])

n_eval_episodes = 5

state_dim = mdps[0][0].state_dim
action_dim = 1
n_actions = mdps[0][0].action_space.n

layers = [l1]
if l2 > 0:
    layers.append(l2)

if not dqn:
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, mdps[0][0].gamma, state_dim, action_dim)
    # Create Q Function
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
else:
    Q, operator = DQN(state_dim, action_dim, n_actions, mdps[0][0].gamma, layers=layers)

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
                 eval_episodes=n_eval_episodes,
                 mean_episodes=mean_episodes,
                 seed=seed,
                 render=render,
                 verbose=verbose)

last_rewards = 5

results = []

if just_one_timestep in range(0, timesteps): # Learn optimal policies just for one timestep
    print("Timestep", just_one_timestep)
    if n_jobs == 1:
        timestep_results = [run(mdp,seed) for mdp in mdps[just_one_timestep]]
    elif n_jobs > 1:
        timestep_results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for mdp in mdps[just_one_timestep])

    results = utils.load_object(sources_file_name) # sources must already exist.
    results[just_one_timestep] = timestep_results  # overwrite
    utils.save_object(results, sources_file_name)

else: # Learn optimal policies for all sources
    for i in range(timesteps):
        
        print("Timestep", i)
        if n_jobs == 1:
            timestep_results = [run(mdp,seed) for mdp in mdps[i]]
        elif n_jobs > 1:
            timestep_results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for mdp in mdps[i])

        results.append(timestep_results)
        utils.save_object(results, sources_file_name)

# Save tasks to file
tasks = mdps[-1]
print("Tasks")
print("Speeds:", [t.get_info()[1] for t in tasks])

utils.save_object(tasks, tasks_file_name)