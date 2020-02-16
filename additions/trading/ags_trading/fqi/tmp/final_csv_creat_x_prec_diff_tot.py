import pandas as pd
import numpy as np

#######################################################

''' --- Real Data --- '''

prec = 60
year = 2017
days_csv = 25
#data_real = '2017-EURUSD BGN Curncy-1m.csv'
data_real = '2017_25_days.csv'
cs_r = pd.read_csv(data_real) # read the csv file
price_op = (cs_r[:]['open']).values
count_ = (cs_r[:]['count']).values
final = None
kk = 1


""" --- CREATE DATASET FQI --- """

for i in range(0,len(cs_r),1230):
    
    print('day = ' + str(kk))
    price = price_op[i:i+1229]
    count = count_[i:i+1229]
    count_drop = count[prec:len(count)-2] # considero solo le righe di interesse da prec in poi
    price_ = price[1:len(price)] # create price_prime column
    price_ = np.append(price_,[np.nan]) # add NaN  at the last value
    time = np.arange(len(price))/(len(price)-1) # extract time column
    time_ = time[1:len(time)] # create time_prime column
    time_ = np.append(time_,[np.nan]) # add NaN at the last value
    done = np.arange(len(price)-2)*0 # create a 0 values column
    done = np.append(done,[1]) # append 1 on the last value
    cs_new = pd.DataFrame({"TIME":time,"PRICE":price,"TIME_p":time_,"PRICE_p":price_}) # create a DataFrame 
    cs_new.to_csv("csv_new.csv",index = False) # save in .csv file
    cs_new = cs_new.drop(cs_new.index[len(cs_new)-1]) # drop last raw (to delete NaN)
    cs_new = cs_new.join(pd.DataFrame({"DONE":done})) # add "DONE" column
    
    
    price_new = (cs_new.values)[:,0]
    price_prec = price_new[(prec-1):(len(price_new)-1)] 
    for i in range(1,prec):
        price_prec = np.column_stack((price_prec, (price_new[(prec-1-i):(len(price_new)-1-i)])))
    
    cs_new = cs_new.drop(cs_new.index[0:prec]) # drop rows from 0 to (prec-1) of cs_new
    p_prec = pd.DataFrame(price_prec) # convert price_prec in DataFrame
    
    cs_new_60 = np.column_stack((cs_new,p_prec)) # ma è un array !!!!!!!!!!!!!!!!
        
    pri = np.column_stack(((cs_new_60)[:,0], (cs_new_60)[:,5:prec+5])) # price + price_precedenti
    dif = []
    for k in range(0,len(pri)):
        dif.append(np.diff(pri[k,:]))
    dif = np.asarray(dif)
    
    cs_new_60 = np.column_stack((cs_new, p_prec, dif)) # differenze (derivate)
    
    # aggiungo price_prec_prime e diff_prime
    price_prec_prime = pri[:,0:prec]
    diff_prime = dif[1:len(dif),:] # occhio: il numero di colonne è più piccolo di 1 colonna
    
    cs_new_60 = np.column_stack((cs_new, p_prec, dif, price_prec_prime))
    cs_new_60 = np.delete(cs_new_60, len(cs_new_60)-1, axis=0) # elimino ultima riga per Diff_prime ultimo
    cs_new_60 = np.column_stack((cs_new_60, diff_prime))
    cs_new_60[len(cs_new_60)-1,4] = 1 # metto a 1 L'ULTIMO DONE
    cs_new_60 = np.column_stack((cs_new_60, count_drop)) # aggiungo Count
    cs_new_60 = pd.DataFrame(cs_new_60)
    cs_new_60 = cs_new_60.rename(columns={0: "PRICE", 1: "PRICE_p", 2: "TIME", 3: "TIME_p", 4: "DONE", 245: "Count"})
    
    
    # RENAME columns
    
    for ko in range(5,65):
        #print(kk)
        filename = 'PRICE_(t-' + str(ko-4) + ')'
        cs_new_60 = cs_new_60.rename(columns={ko: filename})
        
    for kj in range(65,125):
        #print(kj)
        filename = 'DIFF_(t-' + str(kj-60-4) + ')'
        cs_new_60 = cs_new_60.rename(columns={kj: filename})
        
    for kl in range(125,185):
        #print(kl)
        filename = 'PRICE_p_(t-' + str(kl-120-4) + ')'
        cs_new_60 = cs_new_60.rename(columns={kl: filename})
        
    for km in range(185,245):
        #print(km)
        filename = 'DIFF_p_(t-' + str(km-180-4) + ')'
        cs_new_60 = cs_new_60.rename(columns={km: filename})
    
            
    final_df = None
    
    for a in [-1, 0, 1]:
        for p in [-1, 0, 1]:
            _df = cs_new_60.copy()
            _df["ACTION"] = a
            _df["PORTFOLIO"] = p
            _df["PORTFOLIO_p"] = a
            if final_df is None:
                final_df = _df
            else:
                final_df = final_df.append(_df)
        
    # REWARD
    final_df['REWARD'] = final_df['PORTFOLIO_p'] * (final_df['PRICE_p'] - final_df['PRICE']) - \
                abs(final_df['PORTFOLIO_p'] - final_df['PORTFOLIO'])*(2/1e5)
    final_df['REWARD'] = final_df['REWARD']*1e3
    
    if final is None:
        final = final_df
    else:
        final = final.append(final_df)
        
    kk=kk+1
        
# save final csv
filename = 'final_'+str(days_csv)+'_days_' + str(year) + '_' + str(prec) + '_prec_diff.csv'
final.to_csv(filename,index = False)

    
