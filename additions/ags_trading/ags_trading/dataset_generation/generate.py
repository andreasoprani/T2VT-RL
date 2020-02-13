import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def generate_full_dataset(input_path, output_path):
    DAY_LENGTH = 1230
    WINDOW_SIZE = 60

    # Read prices from dataset
    cs_r = pd.read_csv(input_path) # read the csv file
    price_op = (cs_r[:]['open']).values
    final = None
    kk = 1

    # Create the dataset for FQI
    for i in range(0, len(cs_r), DAY_LENGTH):
        print('day = ' + str(kk))
        price = price_op[i:i+DAY_LENGTH-1]
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

        price_new = (cs_new.values)[:,1]
        price_prec = price_new[(WINDOW_SIZE-1):(len(price_new)-1)]
        for i in range(1,WINDOW_SIZE):
            price_prec = np.column_stack((price_prec, (price_new[(WINDOW_SIZE-1-i):(len(price_new)-1-i)])))

        cs_new = cs_new.drop(cs_new.index[0:WINDOW_SIZE]) # drop rows from 0 to (prec-1) of cs_new
        p_prec = pd.DataFrame(price_prec) # convert price_prec in DataFrame

        cs_new_60 = np.column_stack((cs_new,p_prec)) # ma è un array !!!!!!!!!!!!!!!!

        #renaming = {k:'PRICE_t-' + str(k) for k in range(1,61)}
        #cs_new_60 = cs_new_60.rename(columns=renaming)

        pri = np.column_stack(((cs_new_60)[:,1], (cs_new_60)[:,5:WINDOW_SIZE+5])) # price + price_precedenti
        dif = []
        for k in range(0,len(pri)):
            dif.append(np.diff(pri[k,:]))
        dif = np.asarray(dif)

        cs_new_60 = np.column_stack((cs_new, p_prec, dif)) # differenze (derivate)

        # aggiungo price_prec_prime e diff_prime
        price_prec_prime = pri[:,0:WINDOW_SIZE]
        diff_prime = dif[1:len(dif),:] # occhio: il numero di colonne è più piccolo di 1 colonna

        cs_new_60 = np.column_stack((cs_new, p_prec, dif, price_prec_prime))
        cs_new_60 = np.delete(cs_new_60, len(cs_new_60)-1, axis=0) # elimino ultima riga per Diff_prime ultimo
        cs_new_60 = np.column_stack((cs_new_60, diff_prime))
        cs_new_60[len(cs_new_60)-1,4] = 1 # metto a 1 L'ULTIMO DONE
        cs_new_60 = pd.DataFrame(cs_new_60)
        cs_new_60 = cs_new_60.rename(columns={0: "PRICE", 1: "PRICE_p", 2: "TIME", 3: "TIME_p", 4: "DONE"})

        #cc = 'final_25_days_60_prec_diff.csv'
        #dat_c = pd.read_csv(cc)

    #   dat_c.rename(columns={5: 'PRICE_(t-1)'})
        #days_real = 1230

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

        # check che l'ultimo prezzo precedente (quindi pr(t-N)) sia uguale a price[0] di price iniziale
        cs_new_60.iloc[0,WINDOW_SIZE-1] == price[0]


        #price_diff = []
        #for j in range(0,len(price_prec)):
        #    price_diff[j,:] = np.diff(price_prec[j,:])

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


        #plt.plot(final_df['DONE'])

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
    final.to_csv("final_25_days_60_prec_diff.csv",index = False)

    #end = time.time()
    #print(end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='Starting dataset.', type=str, default='../2017_25_days.csv')
    parser.add_argument('--output', help='Path to write the file.', type=str, default='../2017.csv')
    args = parser.parse_args()

    generate_full_dataset(args.input, args.output)
