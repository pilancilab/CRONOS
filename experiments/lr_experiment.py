import jax.numpy as jnp
from training.train import train 
import numpy as np

def lr_grid_exp_fun(problem_data: dict,
                     model_params: dict, 
                     opt_params: dict,
                     task: str, 
                     learning_rate_grid: jnp.ndarray):
    """Given training/test data with a specified model and optimizer, trains the model for different learning rates specifed in learning_rate_grid 
    
    problem_data: dictionary containing training and test matrices along with corresponding targets
    model_params: dict containing parameters that specify the neural network model
    optimizer_params: dict specifying optimizer and other hyperparameters aside from the learning_rate
    learning_rate_grid: grid for which we search over the learning rate,
    
    """


    grid_size = learning_rate_grid.shape[0]
    n_epoch = opt_params['n_epoch']
    if opt_params['optimizer'] == 'Cronos_AM':
        n_epoch = opt_params['n_epoch']
        batch_size = opt_params['batch_size']
        iters_in_epoch = np.ceil(problem_data['training_y'].shape[0]/batch_size).astype(int)
        max_iters = np.round(iters_in_epoch*n_epoch).astype(int)
        cols = np.ceil(0.5*max_iters/opt_params['checkpoint']+1).astype(int)
    else:
        cols = n_epoch+1    
    train_loss = np.zeros((grid_size, cols))
    test_loss = np.zeros((grid_size, cols))
    train_acc = np.zeros((grid_size, cols))
    test_acc = np.zeros((grid_size ,cols))
    times = np.zeros((grid_size ,cols))
    


    for i in range(grid_size):
        opt_params['lr'] = learning_rate_grid[i]
        _, _, perf_log, time_log, _ = train(problem_data['training_X'],  problem_data['training_y'],
                                       problem_data['test_X'], problem_data['test_y'], 
                                       model_params, opt_params, task)
  
        train_loss[i, :] = np.array(perf_log['train_loss'])
        test_loss[i, :] = np.array(perf_log['test_loss'])
        train_acc[i, :] = np.array(perf_log['train_acc'])
        test_acc[i, :] = np.array(perf_log['test_acc'])
        times[i, :] = np.array(time_log['iteration_times'])
    
    best_lr_idx = np.unravel_index(np.argmax(test_acc, axis=None), test_acc.shape)[0]
    best_lr = learning_rate_grid[best_lr_idx]
    metrics = dict(train_loss = train_loss, test_loss = test_loss, train_acc = train_acc, test_acc = test_acc,
                   times = times, best_lr_idx = best_lr_idx, lr_grid = learning_rate_grid)
    
    
    return metrics

def lr_random_search(problem_data:dict,
                     model_params:dict,
                     optimizer_params: dict,
                     task,
                     pow_lr_min,
                     pow_lr_max,
                     grid_size,
                     tuning_seed):
    rng = np.random.default_rng(seed=tuning_seed)
    lr_grid = rng.uniform(pow_lr_min, pow_lr_max, (grid_size,))
    lr_grid = 10**(lr_grid)
    return lr_grid_exp_fun(problem_data, model_params, optimizer_params, task, lr_grid)
              
