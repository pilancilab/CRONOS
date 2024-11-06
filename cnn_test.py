import jax
import numpy as np
import os
from experiments.lr_experiment import lr_grid_exp_fun, lr_random_search
from plotting.metric_plotting import plot_median_metric
from utils.load_data import load_cifar, load_food, load_imagenet, load_imagenet512
import pickle

# manually change the following variables DATASET and MODEL
DATASET = 'imgnet171' # change to 'food', 'imgnet171', 'imgnet'
MODEL = 'cnn' # change to 'mlp', 'cnn', 'gpt2'
OUTPUT_DIR = '/home/miria/Downloads/ZACH/results/' # can change to relative directory 



# Load data

# change seeds for datasets 
# change name of pickle file for each dataset
if DATASET == 'cifar10':
  seed_offset = 0
  classes = (3, 5) # Cat vs. Dog
  training_X, training_y, test_X, test_y = load_cifar(classes)

  training_X = training_X.reshape(10000, 32, 32, 3)
  test_X = test_X.reshape(2000, 32, 32, 3)
  
  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}
  admm_params = dict(rank = 10, beta = 0.001, gamma_ratio = 1, 
  admm_iters = 5, pcg_iters = 5, check_opt = False)
  cronos_params = dict(P_S = 8, rho = 1, beta = 0.001,
                       admm_params = admm_params)
  cronos_am_params = dict(lr = 10**0 ,gamma = 0, n_epoch = 50,
                           cronos_params = cronos_params, 
                           batch_size=1024, checkpoint=10, optimizer = 'Cronos_AM')

elif DATASET == 'food':
  seed_offset = 100
  training_X, training_y, test_X, test_y = load_food()
  
  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}
  admm_params = dict(rank = 10, beta = 0.001, gamma_ratio = 1, 
  admm_iters = 5, pcg_iters = 5, check_opt = False)
  cronos_params = dict(P_S = 8, rho = 1, beta = 0.001,
                       admm_params = admm_params)
  cronos_am_params = dict(lr = 10**0 ,gamma = 0, n_epoch = 50,
                           cronos_params = cronos_params, 
                           batch_size=1024, checkpoint=3, optimizer = 'Cronos_AM')
  training_X = training_X.reshape(3000, 256, 256, 3)
  test_X = test_X.reshape(1000, 256, 256, 3)

elif DATASET == 'imgnet171':
  seed_offset = 200
  training_X, training_y, test_X, test_y = load_imagenet()

  training_X = training_X.reshape(2600, 171, 171, 3)
  test_X = test_X.reshape(100, 171, 171, 3)

  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}
  admm_params = dict(rank = 10, beta = 0.001, gamma_ratio = 1, 
  admm_iters = 5, pcg_iters = 5, check_opt = False)
  cronos_params = dict(P_S = 8, rho = 1, beta = 0.001,
                       admm_params = admm_params)
  cronos_am_params = dict(lr = 10**0 ,gamma = 0, n_epoch = 50,
                           cronos_params = cronos_params, 
                           batch_size=1300, checkpoint=2, optimizer = 'Cronos_AM')

elif DATASET == 'imgnet':
  seed_offset = 300
  training_X, training_y, test_X, test_y = load_imagenet512()

  # Setup optimizers
  opts = {'Adam': dict(optimizer='Adam', n_epoch=50, batch_size=1024), 
  'AdamW': dict(optimizer='AdamW', gamma=10**-4,  n_epoch=50, batch_size=1024), 
  'SGD': dict(optimizer='SGD', momentum=0.9, n_epoch=50, batch_size=1024),
  'Shampoo': dict(optimizer='Shampoo', n_epoch=50, batch_size=1024),  
  'Yogi': dict(optimizer = 'Yogi', n_epoch=50, batch_size=1024)}
  
  admm_params = dict(rank = 10, beta = 0.001, gamma_ratio = 1, 
  admm_iters = 5, pcg_iters = 5, check_opt = False)
  cronos_params = dict(P_S = 8, rho = 1, beta = 0.001,
                       admm_params = admm_params)
  cronos_am_params = dict(lr = 10**0 ,gamma = 0, n_epoch = 50,
                           cronos_params = cronos_params, 
                           batch_size=60, checkpoint=10, optimizer = 'Cronos_AM')

problem_data = dict(training_X=training_X, training_y=training_y, test_X=test_X, test_y=test_y)

# Specify model and task
model_params = dict(type = 'cnn')
task = 'classification'

seeds = [2022 + seed_offset , 2023 + seed_offset, 2024 + seed_offset]
for seed in seeds:
  filename = f"{DATASET}_{MODEL}_seed_{seed}.pkl"
   
  optimizer_metrics = {}
  
  # Parameters for random search
  l, u = -3.5, -1.5
  grid_size = 5
  tuning_seed = 10

  i = 0
  for opt in opts:
    #Set seed
    opts[opt]['seed'] = jax.random.key(seed)
    optimizer_metrics[opt] = lr_random_search(problem_data, model_params, opts[opt], 
                                                 task, l, u, grid_size, tuning_seed+i)
    print("Finished tuning" + " " + opt + "!" )
    i+=1

# CronosAM
  # Set seed and get learning rate
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