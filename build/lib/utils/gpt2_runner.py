import jax
import numpy as np
import jax.numpy as jnp
from utils.gpt2_dataloader import load_data
from models.cvx_relu_mlp import CVX_ReLU_MLP
from optimizers.cronos import admm
from experiments.lr_experiment import lr_random_search
import os
import pickle

def run(model_name, num_batches, cronos_params, adamW_params, opt_seed, data_seed, output_dir):

    Atr, ytr, Atst, ytst, ntr, ntst = load_data(model_name, num_batches, data_seed)

    ##### CRONOS #####

    num_neurons = 10

    #  Setup model

    model = CVX_ReLU_MLP(Atr, ytr, num_neurons, cronos_params['beta'], cronos_params['rho'], jax.random.key(0))

    model.init_model()
    model.Xtst = Atst
    model.ytst = ytst

    print('Training model with CRONOS')

    # Run twice to get compiled version 
    for i in range(2):
        _ , metrics = admm(model, cronos_params)
        if i == 1:
            print('Finished training with CRONOS')
    
    train_peak = np.max(metrics['train_acc'])
    test_peak = np.max(metrics['test_acc'])
    print(f"Peak train accuracy: {train_peak}")
    print(f"Peak test accuracy: {test_peak}")
   
   ##### AdamW #####

    seed_offset = 10
    seeds = [opt_seed + seed_offset , opt_seed + seed_offset, opt_seed + seed_offset]
    problem_data = dict(training_X=Atr, training_y=ytr, test_X=Atst, test_y=ytst)
    for seed in seeds:
        filename = model_name+f"_seed_{seed}.pkl"
    

        optimizer_metrics = {}
        optimizer_metrics['CRONOS'] = metrics
        
        # Parameters for random search
        l, u = -6, -2.5
        grid_size = 8
        tuning_seed = 0

        i = 0
        #adamW_params = dict(optimizer='AdamW', gamma=10**-4,  n_epoch=30, batch_size=1024)
        model_params = dict(type = 'two_layer_mlp')
        task = 'classification'
        adamW_params['seed'] = jax.random.key(seed)
        optimizer_metrics['AdamW'] = lr_random_search(problem_data, model_params, adamW_params, 
                                                        task, l, u, grid_size, tuning_seed+i)
        i+=1
        
        print(np.max(optimizer_metrics['AdamW']['test_acc']))
        
        print(f"Finished running AdamW for seed_{seed}!" )

        # Create the subfolder path
        model_dir = os.path.join(output_dir, model_name)

    # Create the subfolder if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    # Define the full path for the pickle file
    # CHECK filename correctly defined #########################################################################
        pickle_file_path = os.path.join(model_dir, filename)

    # Save the pickle file to the specified directory
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(optimizer_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    