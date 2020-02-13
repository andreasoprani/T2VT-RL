import gym

gym.envs.register(
     id='TradingNew-v0',
     entry_point='ags_trading.trading_env.base:TradingMain',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m-short.csv',
     }
)

gym.envs.register(
     id='TradingPrices-v0',
     entry_point='ags_trading.trading_env.prices:TradingPrices',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingPrices-v1',
     entry_point='ags_trading.unpadded_trading_env.prices:TradingPrices',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingDer-v1',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

# 2017
gym.envs.register(
     id='TradingDer-v3',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
        'fees': 1e-6,
     }
)

gym.envs.register(
     id='TradingDer2018-v1',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2018-EURUSD-BGN-Curncy-1m.csv',
     }
)

# 2018 non-sequential
gym.envs.register(
     id='TradingDer2018-v2',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2018-EURUSD-BGN-Curncy-1m.csv',
        'fees': 1e-6,
     }
)

# 2016
gym.envs.register(
     id='TradingDer2016-v2',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2016-EURUSD_BGN_Curncy-1m.csv',
        'fees': 1e-6,
     }
)

# 2015
gym.envs.register(
     id='TradingDer2015-v2',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2015-EURUSD_BGN_Curncy-1m.csv',
        'fees': 1e-6,
     }
)

# 2014
gym.envs.register(
     id='TradingDer2014-v2',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2014-EURUSD_BGN_Curncy-1m.csv',
        'fees': 1e-6,
     }
)

# 2018 sequential
gym.envs.register(
     id='TradingDer2018-v3',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2018-EURUSD-BGN-Curncy-1m.csv',
        'fees': 1e-6,
        'testing': True,
     }
)

gym.envs.register(
     id='TradingDer2016-v1',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2018-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingDer-v2',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivativesWithStateReward',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingDertest-v1',
     entry_point='ags_trading.unpadded_trading_env.derivatives:TradingDerivatives',
     kwargs={
        'old_csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
        'csv_path': '2018-EURUSD-BGN-Curncy-1m.csv',
     }
)


gym.envs.register(
     id='TradingDer-v0',
     entry_point='ags_trading.trading_env.derivatives:TradingDerivatives',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingDerDoubleState-v0',
     entry_point='ags_trading.trading_env.derivatives:TradingDerivativesWithStateReward',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m.csv',
     }
)

gym.envs.register(
     id='TradingNorm-v0',
     entry_point='ags_trading.trading_env.normed_prices:TradingNormedPrices',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m-short.csv',
     }
)

gym.envs.register(
     id='VecTradingNew-v0',
     entry_point='ags_trading.vectorized_trading_env.base:VecTradingMain',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m-short.csv',
     }
)

gym.envs.register(
     id='VecTradingDer-v0',
     entry_point='ags_trading.vectorized_trading_env.derivatives:VecTradingDerivatives',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m-short.csv',
     }
)

gym.envs.register(
     id='VecTradingNorm-v0',
     entry_point='ags_trading.vectorized_trading_env.normed_prices:VecTradingNormedPrices',
     kwargs={
        'csv_path': '2017-EURUSD-BGN-Curncy-1m-short.csv',
     }
)