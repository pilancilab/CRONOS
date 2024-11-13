'''This runner for vision exp contains param serach
   (in-depth hyperparameter tuning)
'''

# import jax
# import numpy as np
# import os
# from experiments.lr_experiment import lr_grid_exp_fun, lr_random_search
# from plotting.metric_plotting import plot_median_metric
# from utils.load_data import load_cifar, load_food, load_imagenet, load_imagenet512
# import pickle


# # manually change the following variables DATASET and MODEL
# DATASET = 'imagenet512' # cifar10, food, imagenet, imagenet512
# MODEL = 'mlp' # change to 'mlp', 'cnn', gpt experiments have separate runner
# OUTPUT_DIR = '/home/miria/VISION_TUES/' # can change to relative directory 


import jax
import sys
import numpy as np
import os
from experiments import lr_grid_exp_fun, lr_random_search
from utils import load_cifar, load_food, load_imagenet, load_imagenet512
#from plotting.metric_plotting import plot_median_metric

import pickle
from config import *

# Load data

# change seeds for datasets 
# change name of pickle file for each dataset
if DATASET == 'cifar10':
  seed_offset = 0
  classes = (0, 1) # Automobile vs. Airplane
  training_X, training_y, test_X, test_y = load_cifar(classes)
  
  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}

elif DATASET == 'food':
  seed_offset = 100
  training_X, training_y, test_X, test_y = load_food()
  
  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}

elif DATASET == 'imagenet':
  seed_offset = 200
  training_X, training_y, test_X, test_y = load_imagenet()

  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=600), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=600), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=600),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=600),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=600)}

elif DATASET == 'imagenet512':
  seed_offset = 300
  training_X, training_y, test_X, test_y = load_imagenet512()

  #Setup optimizers
  # opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=600), 
  # 'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=600), 
  # 'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=600),
  # 'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=600),  
  # 'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=600)}
  opts = {}






for rho in [10**(-x) for x in range(2, 5)]:  # rho from 0.01 to 0.000001
    for admm_iters in range(4, 8):  # admm_iters between 4 and 20
        for pcg_iters in range(25, 45):  # pcg_iters between 28 and 50

          admm_params = dict(rank = 10, beta = 0.001, gamma_ratio = 1, 
          admm_iters = admm_iters, pcg_iters = pcg_iters, check_opt = False)
          cronos_params = dict(P_S = 10, rho = rho, beta = 0.001,
                                admm_params = admm_params)
          cronos_am_params = dict(lr = 10**0 ,gamma = 0, n_epoch = 50,
                                    cronos_params = cronos_params, 
                                    batch_size=600, checkpoint=1, optimizer = 'Cronos_AM')

          problem_data = dict(training_X=training_X, training_y=training_y, test_X=test_X, test_y=test_y)

          # Specify model and task
          model_params = dict(type = 'relu-mlp')
          task = 'classification'

          seeds = [2022 + seed_offset , 2023 + seed_offset, 2024 + seed_offset]
          for seed in seeds:
            #filename = f"{DATASET}_{MODEL}_seed_{seed}_vmapstack.pkl"
            filename = f"{DATASET}_{MODEL}_rho_{rho}_admm_{admm_iters}_pcg_{pcg_iters}_seed_{seed}.pkl"
            
            optimizer_metrics = {}
            
            # Parameters for random search
            l, u = -6, -2 # l=-5.5, u = -2
            grid_size = 10 # prev 5
            tuning_seed = 0

            i = 0
            for opt in opts:
              opts[opt]['seed'] = jax.random.key(seed)
              optimizer_metrics[opt] = lr_random_search(problem_data, model_params, opts[opt], 
                                                          task, l, u, grid_size, tuning_seed+i)
              print("Finished tuning" + " " + opt + "!" )
              i+=1

          # CronosAM
            cronos_am_params['seed'] = jax.random.key(seed)
            cronos_params['seed'] = jax.random.key(seed)
            lr = cronos_am_params['lr']
            for i in range(2):
              optimizer_metrics['Cronos_AM'] = lr_grid_exp_fun(problem_data, model_params, cronos_am_params, 
              task, np.array([lr]))
            
            # Create the subfolder path
            model_dir = os.path.join(OUTPUT_DIR, MODEL)

            # Create the subfolder if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)

            # Define the full path for the pickle file
            # CHECK filename correctly defined #########################################################################
            pickle_file_path = os.path.join(model_dir, filename)

            # Save the pickle file to the specified directory
            with open(pickle_file_path, 'wb') as handle:
                pickle.dump(optimizer_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)