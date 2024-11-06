import numpy as np
import os
import pickle 

def get_best_acc(seeds, dataset, model_type, metric):
    master_folder_path = os.getcwd()
    best_metric = {}
    for opt in ['SGD', 'Adam', 'AdamW', 'Yogi', 'Shampoo', 'Cronos_AM'] :
        best_metric[opt] = 0
    for seed in seeds:
    # Load pickle file
        subfolder = 'results'
        filename = f'{dataset}'+'_'+f'{model_type}'+f'_seed_{seed}.pkl'.format(seed)
        pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
        with open(pickle_file_path, 'rb') as handle:
            results = pickle.load(handle)
            for opt in results:
                if metric in {'train_acc', 'test_acc'}:
                    best_metric[opt]+= np.max(results[opt][metric])/len(seeds)
                else:
                    best_metric[opt] = np.min(results[opt][metric])/len(seeds)
    return best_metric

def get_avg_time(seeds, dataset, model_type):
    master_folder_path = os.getcwd()
    mean_epoch_times = {}
    for opt in ['SGD', 'Adam', 'AdamW', 'Yogi', 'Shampoo', 'Cronos_AM']:
        mean_epoch_times[opt] = 0

    for seed in seeds:
        # Load pickle file
        subfolder = 'results'
        filename = f'{dataset}'+'_'+f'{model_type}'+f'_seed_{seed}.pkl'.format(seed)
        pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
        with open(pickle_file_path, 'rb') as handle:
            results = pickle.load(handle)
            
        for opt in results:
            mean_epoch_times[opt] += np.mean(results[opt]['times'])/3
    
    return mean_epoch_times



    
