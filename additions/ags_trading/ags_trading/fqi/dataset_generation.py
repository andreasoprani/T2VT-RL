"""
    Generate the extended dataset for FQI.
    TODO
"""

import pandas as pd
import numpy as np
import logging

def generate(source=None, lag=60, use_prices=True, use_derivatives=True,
                reward_scale=1e3, fee=2/1e5):
    assert source is not None, 'Data source must be given.'
    logging.info('Starting...')

    # ======== Data loading ========
    data = pd.read_csv(source)
    logging.info('Loaded dataset: %s' % (data.shape,))

    # ======== Loop over days ========
    days = data['day'].unique()
    dfs = []
    for day in days:
        # Create a new df only for the day with the open position and the volatility
        day_df = data[data['day'] == day][['open', 'count', 'day']].reset_index(drop=True)
        # Add the time column (index of time of day)
        day_df['time_index'] = day_df.index
        day_df['time'] = day_df.index / day_df.shape[0]
        day_opening = day_df['open'].iloc[0]
        # Create lag columns, also with t+1 (next state)
        if use_prices:
            for i in range(-1, lag):
                day_df['P' + str(i)] = day_df['open'].shift(i) #- day_opening
        if use_derivatives:
            for i in range(-1, lag):
                day_df['D' + str(i)] = day_df['open'].shift(i) - day_df['open'].shift(i+1)
        # Remove nan row (last timestep)
        day_df = day_df.dropna(subset=['P-1'])
        # Fill lagged rows with zeros
        day_df = day_df.fillna(0)
        # Set done flag
        day_df['done'] = 0
        day_df.loc[day_df.index[-1], 'done'] = 1
        # Append to list of partial results
        dfs.append(day_df)
    df_lagged = pd.concat(dfs)
    logging.info('Generated lagged prices and derivatives: %s' % (df_lagged.shape,))

    # ======== Add portfolio and actions ========
    generated_df = []
    for portfolio in [-1, 0, 1]:
        for action in [-1, 0, 1]:
            _df = df_lagged.copy()
            _df['portfolio'] = portfolio
            _df['action'] = action
            generated_df.append(_df)
    generated_df = pd.concat(generated_df)
    logging.info('Generated expanded dataset: %s' % (generated_df.shape,))

    # ======== Compute reward ========
    generated_df['reward'] = (generated_df['action'] * generated_df['D-1']) - \
                             abs(generated_df['action'] - generated_df['portfolio'])*fee
    generated_df['reward'] *= reward_scale
    logging.info('Computed the reward.')

    return generated_df


if __name__ == '__main__':
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data sources
    parser.add_argument('--source', help='CSV containing the prices.', type=str, default=None)
    parser.add_argument('--output', help='Output filename.', type=str, default=None)
    # Parameters
    parser.add_argument('--lag', help='Number of previous prices to include.', type=int, default=60)
    parser.add_argument('--fee', help='Fee for each transaction.', type=float, default=2/1e5)
    parser.add_argument('--reward_scale', help='Scale the reward for computational issues.', type=float, default=1e3)
    parser.add_argument('--log', help='Log level', default='INFO')
    args = parser.parse_args()
    # Logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    assert isinstance(numeric_level, int), 'Invalid log level: %s' % loglevel
    logging.basicConfig(level=numeric_level, format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    args = vars(args)
    del args['log']
    output_name = args['output']
    del args['output']
    assert output_name is not None, 'Provide an output filename.'
    # Call the train function with arguments
    output_df = generate(**args)
    # Saving
    output_df.to_csv(output_name)
    logging.info('Saved dataframe to %s' % (output_name, ))
