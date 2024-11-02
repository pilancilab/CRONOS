import pickle
import os
from metric_plotting import plot_median_metric, plot_best_metric
import matplotlib.pyplot as plt

colors = {'Adam': 'tab:red', 'AdamW': 'tab:blue', 'SGD': 'tab:purple', 'Shampoo': 'brown', 'Yogi': 'tab:pink', 'DAdam': 'black', 'Cronos_AM': 'goldenrod'}
#plt.rcParams['text.usetex'] = False

# ####### CIFAR 10 ####### 
# seeds = [2022, 2023, 2024]
# master_folder_path = '/home/ubuntu/utahfiles/'
# for seed in seeds:
#     # Load pickle file
#     subfolder = 'resultsplots'
#     filename = 'cifar10_mlp_seed_{}.pkl'.format(seed)
#     pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
#     with open(pickle_file_path, 'rb') as handle:
#         metrics = pickle.load(handle)
#     for metric_type in ['med','best']:
#         for metric in ['passes','time']:
#             subfolder = 'cifar10_mlp'
#             subsubfolder = 'seed_{}'.format(seed)
#             pdf_filename = 'cifar10_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
#             os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
#             dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
#             if metric_type == 'med':
#                 plot_median_metric(metrics, 'test_acc', metric, colors, dir,  t_cutoff=1)
#             else:
#                 plot_best_metric(metrics, 'test_acc', metric, colors, dir)

# ####### FOOD ####### 
# seeds = [2122, 2123, 2124]
# master_folder_path = '/home/ubuntu/utahfiles/'
# for seed in seeds:
#     # Load pickle file
#     subfolder = 'resultsplots'
#     filename = 'food_mlp_seed_{}.pkl'.format(seed)
#     pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
#     with open(pickle_file_path, 'rb') as handle:
#         metrics = pickle.load(handle)
#     for metric_type in ['med','best']:
#         for metric in ['passes','time']:
#             subfolder = 'food_mlp'
#             subsubfolder = 'seed_{}'.format(seed)
#             pdf_filename = 'food_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
#             os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
#             dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
#             if metric_type == 'med':
#                 plot_median_metric(metrics, 'test_acc', metric, colors, dir)
#             else:
#                 plot_best_metric(metrics, 'test_acc', metric, colors, dir)

####### ImageNet ####### 
seeds = [2322, 2323, 2324]
master_folder_path = '/home/miria/Desktop/mlp'
for seed in seeds:
    # Load pickle file
    subfolder = 'resultsplots'
    filename = 'imagenet512_mlp_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
    for metric_type in ['med','best']:
        for metric in ['passes','time']:
            subfolder = 'imagenet_mlp'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'imagenet_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            if metric_type == 'med':
                plot_median_metric(metrics, 'test_acc', metric, colors, dir, t_cutoff=6)
            else:
                plot_best_metric(metrics, 'test_acc', metric, colors, dir) #t_cutoff = 6)

# ####### ImageNet171 ####### 
# seeds = [2222, 2223, 2224]
# master_folder_path = '/home/ubuntu/utahfiles/'
# for seed in seeds:
#     # Load pickle file
#     subfolder = 'resultsplots'
#     filename = 'imgnet171_mlp_seed_{}.pkl'.format(seed)
#     pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
#     with open(pickle_file_path, 'rb') as handle:
#         metrics = pickle.load(handle)
#     for metric_type in ['med','best']:
#         for metric in ['passes','time']:
#             subfolder = 'imagenet171_mlp'
#             subsubfolder = 'seed_{}'.format(seed)
#             pdf_filename = 'imagenet171_'+metric_type+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
#             os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
#             dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
#             if metric_type == 'med':
#                 plot_median_metric(metrics, 'test_acc', metric, colors, dir)
#             else:
#                 plot_best_metric(metrics, 'test_acc', metric, colors, dir)
