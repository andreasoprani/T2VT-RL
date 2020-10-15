import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../.."))

import numpy as np
import pandas as pd
from additions.lake.lakeEnv import LakeEnv
from additions.lake.lakecomo import Lakecomo
from additions.approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from additions.algorithms.nt_lake import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.dqn import DQN

# Global parameters
render = False
verbose = True

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_iter", default=1500000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
#parser.add_argument("--exploration_fraction", default=0.5)
#parser.add_argument("--eps_start", default=1.0)
#parser.add_argument("--eps_end", default=0.01)
parser.add_argument("--exploration_fraction", default=0.666666)
parser.add_argument("--eps_start", default=0.5)
parser.add_argument("--eps_end", default=20)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=30)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=32)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--dqn", default=False)

parser.add_argument("--years_per_task", default=12) # Number of years in each task
parser.add_argument("--seeds_per_task", default=1)
parser.add_argument("--just_one_timestep", default=-1) # Used to re-train for just one timestep. -1 = False, 0 -> (timesteps - 1) = timestep to re-train 
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
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
dqn = bool(args.dqn)

years_per_task = int(args.years_per_task)
seeds_per_task = int(args.seeds_per_task)
just_one_timestep = int(args.just_one_timestep)
sources_file_name = str(args.sources_file_name)
tasks_file_name = str(args.tasks_file_name)

# Seed to get reproducible results
seed = 1
np.random.seed(seed)

como_data = pd.read_csv(path + '/../../lake/data/como_data.csv')
demand = np.loadtxt(path + '/../../lake/data/comoDemand.txt')
min_env_flow = np.loadtxt(path + '/../../lake/data/MEF_como.txt')

start_year = int(como_data["year"].head(1))
end_year = int(como_data["year"].tail(1))
l = list(range(start_year, end_year + 1))
chunked_years = [l[i:i + years_per_task] for i in range(0, len(l), years_per_task)]
if len(chunked_years[-1]) < years_per_task:
    chunked_years.pop()
    
tasks_data = []
for ys in chunked_years:
    tasks_data.append(como_data.loc[como_data['year'].isin(ys)])
    
temp_lake = Lakecomo(None, None, min_env_flow, None, None, seed=seed)
temp_inflow = list(como_data.loc[como_data['year'] == 1946, 'in'])
temp_mdp = LakeEnv(temp_inflow, demand, temp_lake)

n_eval_episodes = 5

state_dim = temp_mdp.observation_space.shape[0]
action_dim = 1
n_actions = temp_mdp.N_DISCRETE_ACTIONS

layers = [l1]
if l2 > 0:
    layers.append(l2)

if not dqn:
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, temp_mdp.gamma, state_dim, action_dim)
    # Create Q Function
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
else:
    Q, operator = DQN(state_dim, action_dim, n_actions, temp_mdp.gamma, layers=layers)

def run(data, actions_report_file, seed=None):
    return learn(Q,
                 operator,
                 data,
                 demand,
                 min_env_flow,
                 actions_report_file=actions_report_file,
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

seeds = [9, 44, 404, 240, 259]

last_rewards = 5

results = []

if just_one_timestep in range(0, len(tasks_data) - 1): # Learn optimal policies just for one timestep
    print("Timestep", just_one_timestep)
    if n_jobs == 1:
        timestep_results = [run(tasks_data[just_one_timestep], seeds[j]) for j in range(seeds_per_task)]
    elif n_jobs > 1:
        timestep_results = Parallel(n_jobs=n_jobs)(delayed(run)(tasks_data[just_one_timestep], seeds[j]) for j in range(seeds_per_task))

    results = utils.load_object(sources_file_name) # sources must already exist.
    results[just_one_timestep] = timestep_results  # overwrite
    utils.save_object(results, sources_file_name)

else: # Learn optimal policies for all sources
    for i in range(len(tasks_data) - 1):
        actions_report_file = path + "/../../../actions_report_n=" + str(l1) + "_t=" + str(i) + "_s="
        print("Timestep", i)
        if n_jobs == 1:
            timestep_results = [run(tasks_data[i], actions_report_file + str(j) + ".csv", seeds[j]) for j in range(seeds_per_task)]
        elif n_jobs > 1:
            timestep_results = Parallel(n_jobs=n_jobs)(delayed(run)(tasks_data[i], actions_report_file + str(j) + ".csv", seeds[j]) for j in range(seeds_per_task))

        results.append(timestep_results)
        utils.save_object(results, sources_file_name)

# Save tasks to file

utils.save_object(tasks_data[-1], tasks_file_name)