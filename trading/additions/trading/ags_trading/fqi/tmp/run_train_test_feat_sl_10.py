from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.utilities.evaluation import evaluate_policy
from trlib.algorithms.reinforcement.fqi import FQI
from trading_env_60_discrete import Trading60Discrete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle

# FIXME: ricordarsi di cambiare i parametri iniziali negli environ e anche il nome del csv ###############

start = time.time()
#fold = 5
train_days = 260
test_days = 260
year_train = 2017
year_test = 2018
csv_train = '../data/2017-EURUSD BGN Curncy-1m.csv'
csv_test = '../data/2018-EURUSD BGN Curncy-1m.csv'
csv_fqi_train = '../data/final_2017_60_prec_diff.csv'
ms = 186

''' --- Dataset after Feat select --- '''

#fs = [5,6,7,9,14,15,16,31,64] # prezzi precedenti da feature selezionate # FIXME:
fs = np.arange(5,65) # tutte le features (60 prezzi precedenti)
#fs = np.arange(65,125) # tutte le features (60 differenze precedenti)
fs_p = fs + 120

cc = csv_fqi_train # FIXME:
dat_fqi = pd.read_csv(cc)
dat_ar = dat_fqi.values
sa = np.column_stack(((dat_fqi['PORTFOLIO']).values, (dat_ar[:,fs]), (dat_fqi['ACTION']).values)) # STATE ACTION
s_prime = np.column_stack(((dat_fqi['PORTFOLIO_p']).values, (dat_ar[:,fs_p]))) # STATE PRIME
r = (dat_fqi['REWARD']).values # REWARD

#dat_fqi = dat_fqi.iloc[:,feat_sel]


""" --- ENVIRONMENTS --- """ # FIXME:
target_mdp_train_1 = Trading60Discrete_train_1(csv_train)
target_mdp_test_1 = Trading60Discrete_test_1(csv_test)

actions = [-1, 0, 1]

""" --- PARAMS --- """

minsplit_opt = ms # crossvalidation # FIXME:
#min_split = (minsplit_opt/(train_days*1167*9)) # FIXME: min split in percentage
min_split = ms
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': min_split,
                    'min_samples_leaf': 1,
                    'n_jobs': 1} # FIXME:

max_iterations = 10 # FIXME:
batch_size = 10


""" --- FQI --- """

################ TRAIN ##################
filename = 'TRAIN_' + str(train_days) + ' days - ' + str(year_train) + ' - ' + str(minsplit_opt)+' ms_'+str(max_iterations)+' it'+'_fs'+str(len(fs))
print(filename)
target_mdp_train = target_mdp_train_1 # FIXME: change mdp
n_days_train = train_days
epsilon = 0
pi = EpsilonGreedy(actions, ZeroQ(), epsilon)

#type(pi)


#dat_ = pd.read_csv('dat_fqi_train_1.csv') # FIXME: change csv
#dat_ar = dat_.values
#r = (dat_['REWARD']).values # REWARD
#s_prime = np.column_stack(((dat_['PORTFOLIO_p']).values, (dat_['TIME_p']).values, (dat_ar[:,185:245]))) # STATE PRIME
absorbing = (dat_fqi['DONE']).values # DONE
#sa = np.column_stack(((dat_['PORTFOLIO']).values, (dat_['TIME']).values, (dat_ar[:,65:125]), (dat_['ACTION']).values)) # STATE ACTION

algorithm = FQI(target_mdp_train, pi, verbose = True, actions = actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, **regressor_params)

#mean_scores_train=[]
#actions_train=[]


for i in range(max_iterations):
    algorithm._iter(sa, r, s_prime, absorbing)
    #ris_train,act_train = evaluate_policy(n_days_train, target_mdp_train, pi_tmp, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1)
    #print(ris_train)
    #print(act_train)
    #mean_scores_train.append(ris_train)
    #actions_train.append(act_train)

# Save Model after Train
filename = 'pi_'+ str(train_days) + '_days_' + str(year_train) + '-' + str(minsplit_opt)+'ms_'+str(max_iterations)+'it_fs'+str(len(fs))+'.pkl'
pickle.dump(pi, open(filename,'wb'))


################ TEST ################
pi_fqi = pickle.load(open(filename,'rb'))
##
filename =  'TEST_' + str(test_days) + ' days - ' + str(year_test) + ' - ' + str(minsplit_opt)+' ms_'+str(max_iterations)+' it'
print(filename)
target_mdp_test = target_mdp_test_1 # FIXME: change mdp
mean_scores_test=[]
#actions_test=[]
##
ris_test,act_test = evaluate_policy(test_days, target_mdp_test, pi_fqi, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1)
mean_scores_test.append(ris_test)
mean_scores_test = np.transpose(np.reshape(mean_scores_test, (1, test_days)))
#actions_test.append(act_test)
#act_test_day = np.reshape(actions_test, (test_days, 1167))

m = np.zeros((test_days,1167))
m.fill(np.nan)
for i in range(0,test_days):
    for j in range(0,len(act_test[0,i])):
        m[i,j] = int((act_test[0,i])[j])
act_test_day = m

# EPISODE_LEN
episod_len = []
for i in range(0,25):
    episod_len.append(len(act_test[0,i]))


# SAVE Results to .csv
filename = 'Scor_test-'+ str(test_days) + '_days_' + str(year_test) + '-' +str(max_iterations)+'it_' + str(minsplit_opt)+'ms__fs'+str(len(fs))+'.csv'
(pd.DataFrame(mean_scores_test)).to_csv(filename, index = False)

filename = 'Act_test-'+ str(test_days) + '_days_' + str(year_test) + '-' +str(max_iterations)+'it_' + str(minsplit_opt)+'ms__fs'+str(len(fs))+'.csv'
(pd.DataFrame(act_test_day)).to_csv(filename, index = False)

filename = 'Episode_len-'+ str(test_days) + '_days_' + str(year_test) + '-' +str(max_iterations)+'it_' + str(minsplit_opt)+'ms__fs'+str(len(fs))+'.csv'
(pd.DataFrame(episod_len)).to_csv(filename, index = False)

end = time.time()
print('time = ' + str(end - start))
