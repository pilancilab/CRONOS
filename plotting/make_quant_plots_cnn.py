import pickle
import os
from metric_plotting import plot_median_metric, plot_best_metric

colors = {'Adam': 'tab:red', 'AdamW': 'tab:blue', 'SGD': 'tab:purple', 'Shampoo': 'brown', 'Yogi': 'tab:pink', 'Cronos_AM': 'goldenrod'}

####### CIFAR 10 ####### 
seeds = [2022, 2023, 2024]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'cifar10_cnn_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
    for metric_type in ['med','best']:
        for metric in ['passes','time']:
            subfolder = 'cifar10_cnn'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'cifar10_cnn_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            if metric_type == 'med':
                plot_median_metric(metrics, 'test_acc', metric, colors, dir)
            else:
                plot_best_metric(metrics, 'test_acc', metric, colors, dir)


####### ImageNet171 ####### 
seeds = [2222, 2223, 2224]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'imgnet171_cnn_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
    for metric_type in ['med','best']:
        for metric in ['passes','time']:
            subfolder = 'imagenet171_cnn'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'imagenet171_cnn_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            if metric_type == 'med':
                plot_median_metric(metrics, 'test_acc', metric, colors, dir)
            else:
                plot_best_metric(metrics, 'test_acc', metric, colors, dir)
