import numpy as np
from joblib import Parallel, delayed

def evaluate_policy(n_days, mdp, policy, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1):
    """
    Evaluates a policy on a given MDP.
    
    Parameters
    ----------
    mdp: the environment to use in the evaluation
    policy: the policy to evaluate
    criterion: either 'discounted' or 'average'
    n_episodes: the number of episodes to generate in the evaluation
    initial_states: either None (i), a numpy array (ii), or a list of numpy arrays (iii)
      - (i) initial states are drawn from the MDP distribution
      - (ii) the given array is used as initial state for all episodes
      - (iii) n_episodes is ignored and the episodes are defined by their initial states
    n_threads: the number of threads to use in the evaluation
    
    Returns
    -------
    The mean of the scores and its confidence interval.
    """
    #n_days = 13
    assert criterion == 'average' or criterion == 'discounted'
    
    if n_threads == 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = [_single_eval(n_days, mdp, policy, criterion, initial_states) for _ in range(n_episodes)]
#        s1,s2,s3=_single_eval(mdp, policy, criterion, initial_states)
#        ac=[]
#        for _ in range(n_episodes):
#            ac.append(s3)
#        print(ac)
    elif n_threads > 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, initial_states) for _ in range(n_episodes))
    elif n_threads == 1 and type(initial_states) is list:
        scores = [_single_eval(n_days, mdp, policy, criterion, init_state) for init_state in initial_states]
    elif n_threads > 1 and type(initial_states) is list:
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, init_state) for init_state in initial_states)
    
    n_episodes = len(initial_states) if type(initial_states) is list else n_episodes
    
    scores, ts, actions = zip(*scores)
    
    scores = np.array(scores)
    actions = np.array(actions)
    
    #print(np.std(scores))
    return scores, actions
    #return np.mean(scores), actions #np.std(scores[:,0]) / np.sqrt(n_episodes), np.mean(scores[:,1]) , scores[:,2]

def _single_eval(n_days, mdp, policy, criterion, initial_state):
    ss = []
    aa = []
    n_days = n_days
    dd = 1
    for _ in range(n_days):
        print('day ' + str(dd))
        score = 0
        gamma = mdp.gamma if criterion == "discounted" else 1
        
        s = mdp.reset(initial_state)
        t = 0
        act=[]
        while t < mdp.horizon: # horizon ????
            #print(t)
            #print(score)
            #print(s)
            a = policy.sample_action(s) ##############
            act.append(a)
            s,r,done,_ = mdp.step(a)
            score += r * gamma**t
            t += 1
            if done:
                break
        #print(score)
        ss.append(score)
        aa.append(act)
        dd += 1
        
    
    return ss if criterion == "discounted" else ss / t, t, aa
    #  return score if criterion == "discounted" else score / t, t, act