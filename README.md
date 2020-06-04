# Time-Variant Variational Transfer for Reinforcement Learning

- [Time-Variant Variational Transfer for Reinforcement Learning](#time-variant-variational-transfer-for-reinforcement-learning)
  - [Requirements](#requirements)
  - [Experiments introduction](#experiments-introduction)
  - [Complete run scripts](#complete-run-scripts)
    - [Two-room](#two-room)
    - [Three-room](#three-room)
    - [Mountaincar](#mountaincar)
  - [Sources generation](#sources-generation)
    - [Two-room](#two-room-1)
    - [Three-room](#three-room-1)
    - [Mountaincar](#mountaincar-1)
  - [Algorithms tests](#algorithms-tests)
    - [Two-room](#two-room-2)
    - [Three-room](#three-room-2)
    - [Mountaincar](#mountaincar-2)
  - [Lambda sensitivity and tuning](#lambda-sensitivity-and-tuning)
  - [Plotting](#plotting)
  - [CSVs generation](#csvs-generation)

This repository is the official implementation of Time-Variant Variational Transfer for Reinforcement Learning (link to be added).

## Requirements

```
python 3.7

gym (by OpenAI) (we used version 0.15.4)
joblib          (we used version 0.14.1)
matplotlib      (we used version 3.1.1)
numpy 1.17.4
scipy           (we used version 1.3.2)
torch (PyTorch) (we used version 1.3.1)
```

Note: the version 1.17.4 of numpy is necessary to open the provided pickle files.

## Experiments introduction

This work relies on the work presented in [Transfer of Value Functions via Variational Methods](http://papers.nips.cc/paper/7856-transfer-of-value-functions-via-variational-methods) and it uses part of the [implementation](https://github.com/AndreaTirinzoni/variational-transfer-rl) of that publication. Our additions were organized in the "additions" folder.

The experiments performed are divided into three environments: ```two-room```, ```three-room``` and ```mountaincar```.

For each environment there are three possible experiment types:
* ```linear```
* ```polynomial```
* ```sin```

For each environment and experiment type we tested the MGVT and T2VT algorithms, both with 1 and 3 post components (4 runs in total).

## Complete run scripts

Complete run files allow to generate sources and run all the experiments with a single python command. We suggest to use these scripts to test the same experiments we performed, the single samples generation and algorithms testing scripts are available too but they necessitate far more parameters.

With ```experiment_type``` we refer to the 3 experiment types cited above (```linear```, ```polynomial``` and ```sin```).

The default values are the ones used in our experiments.

These scripts can be used to generate new sources and new results. However, the results are already provided with the repository.

### Two-room

```
python3 additions/experiments/rooms/run-2r.py --exp_type=experiment_type
```

Parameters:

```
--exp_type (str): experiment type ("linear", "sin" or "polynomial").
--gen_samples (bool): to perform samples generation or not, default=False, sources are already present.
--max_iter_gen (int): number of iterations for samples generation.
--mgvt_1 (bool): to perform MGVT with 1 posterior component or not, default=True.
--mgvt_3 (bool): to perform MGVT with 3 posterior components or not, default=True.
--t2vt_1 (bool): to perform T2VT with 1 posterior component or not, default=True.
--t2vt_3 (bool): to perform T2VT with 3 posterior components or not, default=True.
--max_iter (int): number of iterations for algorithms test, -1 for default.
--temporal_bandwidth (float in [0,1]): temporal bandwidth for T2VT, -1 for default=0.3333. Note: in the "sin" experiment type, the t2vt results are stored in different directories depending on the temporal_bandwidth, mgvt doesn't take this detail into consideration, so mgvt results will be stored in the "sin" directory and it should be manually moved in order to plot the results and obtain the csvs.
--load_results (bool): used during testing to augment the number of seeds, default=False.
--n_jobs (int): number of jobs when parallelizing, default=1.
```

### Three-room

```
python3 additions/experiments/rooms/run-3r.py --exp_type=experiment_type
```

Parameters:

```
Same as two-room + 
--no_door_zone (int): distance from the border where no door can be placed (default=2.0).
```

### Mountaincar

```
python3 additions/experiments/mountaincar/run.py --exp_type=experiment_type
```

Parameters as two-room.

## Sources generation

Scripts used for generating the sources and the target tasks.
See the ```gen_samples.py``` scripts for the parameters.

### Two-room

```
python3 additions/experiments/rooms/gen_samples.py --env=two-room-gw --experiment_type=experiment_type ... other params
```

### Three-room

```
python3 additions/experiments/rooms/gen_samples.py --env=three-room-gw --experiment_type=experiment_type ... other params
```

### Mountaincar

```
python3 additions/experiments/mountaincar/gen_samples.py --experiment_type=experiment_type ... other params
```

## Algorithms tests

Algorithm tests are performed for MGVT with ```run_mgvt.py``` scripts and for T2VT with ```run_t2vt.py``` scripts.

See these scripts for the parameters.

### Two-room

```
python3 additions/experiments/rooms/run_mgvt.py --env=two-room-gw --experiment_type=experiment_type --post_components=pc ... other params
python3 additions/experiments/rooms/run_t2vt.py --env=two-room-gw --experiment_type=experiment_type --post_components=pc ... other params
```

### Three-room

```
python3 additions/experiments/rooms/run_mgvt.py --env=three-room-gw --experiment_type=experiment_type --post_components=pc ... other params
python3 additions/experiments/rooms/run_t2vt.py --env=three-room-gw --experiment_type=experiment_type --post_components=pc ... other params
```

### Mountaincar

```
python3 additions/experiments/mountaincar/run_mgvt.py --experiment_type=experiment_type --post_components=pc ... other params
python3 additions/experiments/mountaincar/run_t2vt.py --experiment_type=experiment_type --post_components=pc ... other params
```

## Lambda sensitivity and tuning

We have tested the sensitivity of the parameter lambda and we have provided a heuristic for auto-tuning.  
A grid of lambda values (step=0.1) and the heuristic can be tested (on the two-room and three-room environments) with the following script:

```
python3 additions/experiments/rooms/run_test_lambda.py
```

Parameters:

```
--exp_type (str): experiment type (linear, sin or polynomial), default="": it tests all of them.
--max_iter (int): maximum number of iterations, default=3000 (use 3000 for two-room and 15000 for three-room).
--env (str): two-room-gw or three-room-gw, default=two-room-gw.
--n_runs (int): number of seeds used, default=50.
--load_results (bool): used during testing to augment the number of seeds, default=False.
--n_jobs (int): number of parallel jobs used, default=5.
```

The results will be stored in the correct experiment results path, in the "lambda_test" folder.

## Plotting

Once an experiment is performed, it's possible to plot the results (which reside in the ```results``` folder) with the following command:

```
python3 additions/experiments/plot_results.py --path=results/path/
```

Note: the ```/``` at the end of the path is necessary.

Parameters:

```
--show (bool): show the plots before saving them, default=True.
--title (str): title for the plots, default="".
--lambda_test (bool): alternative plotting for lambda tuning purposes, default=False.
```

If the results of all the experiments are present, it's possible to batch plot them with the following command:

```
python3 additions/experiments/run_all_plots.py
```

## CSVs generation

As with plots, it is possible to generate the csvs of the experiments' results (which reside in the ```results``` folder) with the following command:

```
python3 additions/experiments/gen_csv.py --path=results/path
```

Parameters:

```
--lambda_test (bool): alternative csv for lambda tuning purposes, default=False.
```

Command to generate the csvs in batch:

```
python3 additions/experiments/run_all_csvs.py
```