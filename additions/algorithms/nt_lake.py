import numpy as np
import pandas as pd
from additions.lake.lakeEnv import LakeEnv
from additions.lake.lakecomo import Lakecomo
from misc.policies import EpsilonGreedy, ScheduledEpsilonGreedy, Gibbs, ScheduledGibbs
from additions.approximators.mlp_torch import MLPQFunction
from misc.buffer import Buffer
from misc import utils
import time

from operators.dqn import DQNOperator

def _single_year_eval(mdp, policy, preprocess=lambda x: x):
        
    s = mdp.reset()
    t = 0
    done = False
    score = 0
    
    while not done:
        
        a = policy.sample_action(s)
        s, r, done, _ = mdp.step(a)
        score += r * mdp.gamma ** t
        t += 1

    return score

def learn(Q,
          operator,
          data,
          demand,
          min_env_flow,
          actions_report_file = "",
          max_iter=5000,
          buffer_size=10000,
          batch_size=50,
          alpha=0.001,
          train_freq=1,
          eval_freq=50,
          eps_start=1.0,
          eps_end=0.02,
          exploration_fraction=0.2,
          random_episodes=0,
          eval_states=None,
          eval_episodes=1,
          mean_episodes=50,
          preprocess=lambda x: x,
          seed=None,
          render=False,
          verbose=True):
    
    leap_year_demand = np.insert(demand, 60, demand[59])
    
    if seed is not None:
        np.random.seed(seed)

    # mdp creation
    lake = Lakecomo(None, None, min_env_flow, None, None, seed=seed)
    years = data.year.unique()
    description = str(int(years[0])) + "-" + str(int(years[-1]))
    sampled_year = np.random.choice(years)
    inflow = list(data.loc[data['year'] == sampled_year, 'in'])
    if sampled_year % 4 == 0: # leap years between 1946 and 2011 satisfy this condition even though it's not the complete leap year condition
        mdp = LakeEnv(inflow, leap_year_demand, lake)
    else:
        mdp = LakeEnv(inflow, demand, lake)

    # Randomly initialize the weights in case an MLP is used
    if isinstance(Q, MLPQFunction):
        Q.init_weights()
        if isinstance(operator, DQNOperator):
            operator._q_target._w = Q._w

    # Initialize policies
    schedule = np.linspace(eps_start, eps_end, exploration_fraction * max_iter)
    #pi = ScheduledEpsilonGreedy(Q, np.arange(mdp.N_DISCRETE_ACTIONS), schedule)
    #pi_u = EpsilonGreedy(Q, np.arange(mdp.N_DISCRETE_ACTIONS), epsilon=1)
    #pi_g = EpsilonGreedy(Q, np.arange(mdp.N_DISCRETE_ACTIONS), epsilon=0)
    pi = ScheduledGibbs(Q, np.arange(mdp.N_DISCRETE_ACTIONS), schedule)
    pi_u = Gibbs(Q, np.arange(mdp.N_DISCRETE_ACTIONS), tau=0)
    pi_g = Gibbs(Q, np.arange(mdp.N_DISCRETE_ACTIONS), tau=np.inf)

    # Add random episodes if needed
    init_samples = utils.generate_episodes(mdp, pi_u, n_episodes=random_episodes,
                                           preprocess=preprocess) if random_episodes > 0 else None
    if random_episodes > 0:
        t, s, a, r, s_prime, absorbing, sa = utils.split_data(init_samples, mdp.observation_space.shape[0], mdp.action_dim)
        init_samples = np.concatenate((t[:, np.newaxis], preprocess(s), a, r[:, np.newaxis], preprocess(s_prime),
                                       absorbing[:, np.newaxis]), axis=1)

    # Figure out the effective state-dimension after preprocessing is applied
    eff_state_dim = preprocess(np.zeros(mdp.observation_space.shape[0])).size

    # Create replay buffer
    buffer = Buffer(buffer_size, eff_state_dim)
    n_init_samples = buffer.add_all(init_samples) if random_episodes > 0 else 0

    # Results
    iterations = []
    episodes = []
    n_samples = []
    evaluation_rewards = []
    learning_rewards = []
    episode_rewards = [0.0]
    episode_t = []
    l_2 = []
    l_inf = []

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0

    start_time = time.time()

    if actions_report_file:
        actions_executed = []
        
        columns = list(range(mdp.N_DISCRETE_ACTIONS))
        actions_report_df = pd.DataFrame(columns=columns)
        actions_report_df.to_csv(actions_report_file, index=False)

    done_counter = 0

    # Learning
    for i in range(max_iter):

        # Take epsilon-greedy action wrt current Q-function
        s_prep = preprocess(s)
        a = pi.sample_action(s_prep)
        actions_executed.append(a)
        
        # Step
        s_prime, r, done, _ = mdp.step(a)
        
        # Build the new sample and add it to the dataset
        buffer.add_sample(h, s_prep, a, r, preprocess(s_prime), done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Estimate gradient
            g = operator.gradient_be(Q, buffer.sample_batch(batch_size))
            # Take a gradient step
            Q._w, t, m_t, v_t = utils.adam(Q._w, g, t, m_t, v_t, alpha=alpha)

        # Add reward to last episode
        episode_rewards[-1] += r * mdp.gamma ** h

        s = s_prime
        
        h += 1
        if done or h >= mdp.horizon:
            
            if actions_report_file:
                actions_counts = np.bincount(actions_executed) 
                actions_freqs = list(actions_counts / sum(actions_counts))
                new_row = dict(zip(columns, actions_freqs))
                actions_report_df = actions_report_df.append(new_row, ignore_index=True)
                actions_report_df.to_csv(actions_report_file, index=False)

                actions_executed = []
            
            episode_rewards.append(0.0)
            
            sampled_year = np.random.choice(years) 
            inflow = list(data.loc[data['year'] == sampled_year, 'in'])
            if sampled_year % 4 == 0:
                mdp = LakeEnv(inflow, leap_year_demand, lake)
            else:
                mdp = LakeEnv(inflow, demand, lake)
                
            s = mdp.reset()
                
            h = 0
            episode_t.append(i)
            
            done_counter += 1
            
        # Evaluate model
        if done_counter == eval_freq:

            # Evaluate greedy policy
            scores = []
            for _ in range(eval_episodes):
                sampled_year = np.random.choice(years) 
                inflow = list(data.loc[data['year'] == sampled_year, 'in'])
                if sampled_year % 4 == 0:
                    mdp = LakeEnv(inflow, leap_year_demand, lake)
                else:
                    mdp = LakeEnv(inflow, demand, lake)
                    
                scores.append(_single_year_eval(mdp, pi_g))
            
            rew = np.mean(scores)
            
            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, buffer.sample_batch(batch_size)) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)

            sampled_year = np.random.choice(years)
            inflow = list(data.loc[data['year'] == sampled_year, 'in'])
            
            if sampled_year % 4 == 0:
                mdp = LakeEnv(inflow, leap_year_demand, lake)
            else:
                mdp = LakeEnv(inflow, demand, lake)
                
            s = mdp.reset()

            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = end_time

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} L2 {} L_inf {} time {:.1f} s".format(
                    i, episodes[-1], rew, learning_rew, l_2_err, l_inf_err, elapsed_time))
        
            done_counter = 0
        
        if (i * 100 / max_iter) % 10 == 0:
            print("years:", description, "- Progress:", str(int(i * 100 / max_iter)) + "%")

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, episode_rewards[:len(episode_t)], episode_t]
    weights = np.array(Q._w)

    last_rewards = 5
    print("years:", description, "- Last evaluation rewards:", np.around(evaluation_rewards[-last_rewards:], decimals = 3))

    return [[], weights, run_info]
