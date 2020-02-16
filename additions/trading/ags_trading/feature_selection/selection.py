import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.metrics import r2_score

# REF MIN SPLIT
# 186 = 0.000885

def load_dataset(input_path):
    dat_fqi = pd.read_csv(input_path)
    dat_ar = dat_fqi.values
    reward = (dat_fqi['REWARD']).values # REWARD
    state_action_portfolio = np.column_stack(((dat_fqi['PORTFOLIO']).values, (dat_fqi['TIME']).values, (dat_ar[:,5:65]), (dat_fqi['ACTION']).values)) # STATE ACTION
    portfolio_action = np.column_stack(((dat_fqi['PORTFOLIO']).values, (dat_fqi['ACTION']).values))
    return state_action_portfolio, portfolio_action, reward

def create_model(min_split=186, njobs=1, verbose=False):
    regressor_params = {'n_estimators': 50,
                        'criterion': 'mse',
                        'min_samples_split': min_split,
                        'min_samples_leaf': 1,
                        'n_jobs': njobs,
                        'verbose': verbose}
    model = ExtraTreesRegressor(**regressor_params)
    return model

def compute_r2(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    return y_pred, r2_score(y, y_pred)

def get_best_feature(model, X, y, selected_indexes, y_pred_selected):
    # Compute the delta
    delta = y - y_pred_selected
    model.fit(X, delta)
    # Select best from feature importance
    ff = model.feature_importances_
    dic = dict(enumerate(ff))
    dic_ord = sorted(dic.items(), key = lambda x : x[1], reverse = True)
    R2 = -1
    for k in range(len(dic_ord)):
        if dic_ord[k][0] not in selected_indexes:
            selected_indexes.append(dic_ord[k][0])
            break
    # Compute R2 with new features
    x_sel = X[:,selected_indexes]
    y_pred, R2 = compute_r2(model, x_sel, y)

    return y_pred, R2

def feature_selection(input_path, min_split=186, njobs=1, epsilon=1e-4):

    # Load dataset
    D, pa, r = load_dataset(input_path)
    # Create model
    model = create_model(min_split=min_split, njobs=njobs)
    _, r2_start = compute_r2(model, D, r)
    r_pred_start, r2_portfolio_action = compute_r2(model, pa, r)

    # Init log structures
    R2_log = [r2_portfolio_action]
    selected_indexes = [0, 62] # Features ind Selected
    delta_R2 = eps + 1 # Set to always enter the loop

    while (delta_R2 > eps):
        y_pred, R2 = get_best_feature(model, D, r, selected_indexes, r_pred_start)
        delta_R2 = R2 - R2_log[-1]
        R2_log.append(R2)

    plt.plot(R2_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='Starting dataset.', type=str, default='../2017_25_days.csv')
    parser.add_argument('--min_split', help='Min split for the regressor.', type=int, default=186)
    parser.add_argument('--njobs', help='Njobs for the regressor.', type=int, default=1)
    args = parser.parse_args()

    feature_selection(args.input, min_split=args.min_split)
