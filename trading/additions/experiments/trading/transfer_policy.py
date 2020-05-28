import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

from joblib import Parallel, delayed
import numpy as np
from misc import utils
import matplotlib.pyplot as plt
import torch
import gym
from additions.approximators.mlp_torch import MLPQFunction
from ags_trading.vectorized_trading_env.prices import VecTradingPrices

# params
batch_size = 128
iterations = 100001
alpha = 0.001
layers = [128, 128]
save_freq = 2000
n_actions_plot = 10000

sources_file_name = path + "/sources"
tasks_file_name = path + "/tasks"

dataset = {
    '2015': {
        'data': path + '/dataset-2015',
        'mdp': gym.make('VecTradingPrices2015-v2'),
        'save_path': path + '/all_Qs-2015'
    },
    #'2016': {
    #    'data': path + '/dataset-2016',
    #    'mdp': gym.make('VecTradingPrices2016-v2'),
    #    'save_path': path + '/all_Qs-2016'
    #},
    #'2017': {
    #    'data': path + '/dataset-2017',
    #    'mdp': gym.make('VecTradingPrices-v3'),
    #    'save_path': path + '/all_Qs-2017'
    #},
    #'2018': {
    #    'data': path + '/dataset-2018',
    #    'mdp': gym.make('VecTradingPrices2018-v2'),
    #    'save_path': path + '/all_Qs-2018'
    #}
} 

tasks = [gym.make('TradingDer2018-v2')]

def compute_loss(Q, datapoints):

    error = torch.zeros(len(datapoints))

    for i, dp in enumerate(datapoints):
        error[i] = (torch.tensor(dp[1]) - Q._nn.forward(dp[0])[0]).sum().pow(2)

    return error.mean()

def compute_loss_single_action(Q, datapoints, a):

    error = torch.zeros(len(datapoints))

    for i, dp in enumerate(datapoints):
        
        error[i] = (torch.tensor(dp[1])[a] - Q._nn.forward(dp[0])[a]).pow(2)

    return error.mean()

def compute_gradient(Q, datapoints):

    with torch.enable_grad():
        Q.gradient(prepare=True)
        loss = compute_loss(Q, datapoints)
        loss.backward()
        grad = Q.gradient()

    return loss.detach().numpy(), grad

def compute_gradient_single_action(Q, datapoints, a):

    with torch.enable_grad():
        Q.gradient(prepare=True)
        loss = compute_loss_single_action(Q, datapoints, a)
        loss.backward()
        grad = Q.gradient()

    return loss.detach().numpy(), grad

def plot_actions(dataset_path, qw, index, task, n_actions, save_path):

    dataset = utils.load_object(dataset_path)
    dataset = np.array(dataset)

    actions_etr = np.zeros((n_actions, 3))
    for i in range(n_actions):
        for j in range(3):
            actions_etr[i, j] = dataset[i, 1][j]

    actions_nn = np.zeros((n_actions, 3))

    q = MLPQFunction(task.state_dim, task.action_space.n, layers=layers, initial_params=qw)

    task.starting_day_index = 0
    task.reset()

    actions_counter = 0

    for di in range(task.n_days):

        task.starting_day_index = di
        s = task.reset()

        done = False
        while not done:
            a_list = q.value_actions(s)
            actions_nn[actions_counter, :] = a_list
            a = np.argmax(a_list)
            s, r, done, _ = task.step([a])

            done = done[0]

            actions_counter += 1
            if actions_counter >= n_actions:
                break

            percentage = actions_counter * 100 / n_actions
            if percentage % 10 == 0:
                print("Actions evaluation: {0:3d}%".format(int(percentage)))

        if actions_counter >= n_actions:
            break

    fig, ax = plt.subplots(3, sharex=True, figsize=(16, 9))

    for i in range(3):
        ax[i].plot(actions_etr[:10000, i], label="ETR")
        ax[i].plot(actions_nn[:10000, i], label="NN")
        ax[i].set_title("Action " + str(i-1))
        ax[i].legend()
    
    plt.savefig(save_path + '.pdf', format='pdf')
    

def transfer(dataset_path, mdp, save_path, iterations, year, seed = 0):

    np.random.seed(seed)

    data = utils.load_object(dataset_path)
    data = np.array(data)

    state_dim = mdp.state_dim
    n_actions = mdp.action_space.n 
    mdp.starting_day_index = 0
    mdp.reset()
    day_length = len(mdp.prices[0])

    Q = MLPQFunction(state_dim, n_actions, layers=layers)
    Q.init_weights()
    
    m_t = 0
    v_t = 0
    t = 0

    utils.save_object([], save_path)

    losses = [[], [], []]

    for i in range(iterations):

        # sample time of day
        time = int(np.random.uniform(low=0, high=day_length))
        datapoints = np.arange(0, len(data) - day_length, day_length)
        datapoints += time
        datapoints = data[datapoints]
        np.random.shuffle(datapoints)
        datapoints = datapoints[:batch_size]

        for a in range(n_actions):
            with torch.autograd.set_detect_anomaly(True):
                train_loss, grad = compute_gradient_single_action(Q, datapoints, a)
            
            losses[a].append(train_loss)

            print("Y: {0}, I: {1:5d}, Time: {2:4d}, A: {3:1d}, Grad: {4:8.6f}, Train Loss: {5:8.6f}".format(year, i, time, a, np.linalg.norm(grad), train_loss))
            
            Q._w, t, m_t, v_t = utils.adam(Q._w, grad, t, m_t, v_t, alpha=alpha)

        if save_freq > 0 and i % save_freq == 0:
            past_Qs = utils.load_object(save_path)
            past_Qs.append(np.array(Q._w))
            utils.save_object(past_Qs, save_path)
            plot_actions(dataset_path, Q._w, i, mdp, n_actions_plot, path + "/plot-" + year + "-" + str(i))

    
    print("Model selected index: {0:4d}, Train Loss: [{1:8.6f}, {2:8.6f}, {3:8.6f}]".format(i, losses[0][i], losses[1][i], losses[2][i]))

    return [mdp.get_info(), np.array(Q._w), losses]

results = []

for k, v in dataset.items():
    print(k)
    results.append([transfer(v['data'], v['mdp'], v['save_path'], iterations, k)])
    utils.save_object(results, sources_file_name)

utils.save_object(tasks, tasks_file_name)