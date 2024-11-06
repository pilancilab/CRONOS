import pickle
import os
from metric_plotting import multi_joy_plot

colors = {'Adam': 'tab:red', 'AdamW': 'tab:blue', 'SGD': 'tab:purple', 'Shampoo': 'brown', 'Yogi': 'tab:pink', 'Cronos_AM': 'goldenrod'}

####### CIFAR 10 ####### 
seeds = [2022, 2023, 2024]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'cifar10_mlp_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
   
    for metric in ['passes', 'time']:
        subfolder = 'cifar10_mlp'
        subsubfolder = 'seed_{}'.format(seed)
        pdf_filename = 'cifar10_'+'trajs'+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
        os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
        dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
        multi_joy_plot(metrics, metric, colors, dir, t_cutoff=0.8)

####### FOOD ####### 
seeds = [2122, 2123, 2124]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'food_mlp_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)

    for metric in ['passes','time']:
            subfolder = 'food_mlp'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'food_'+'trajs'+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            multi_joy_plot(metrics, metric, colors, dir)

####### ImageNet ####### 
seeds = [2322, 2323, 2324]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'imagenet512_mlp_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
   
    for metric in ['passes','time']:
            subfolder = 'imagenet_mlp'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'imagenet_'+'trajs'+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            multi_joy_plot(metrics, metric, colors, dir, t_cutoff=0.5)

####### ImageNet171 ####### 
seeds = [2222, 2223, 2224]
master_folder_path = os.getcwd()
for seed in seeds:
    # Load pickle file
    subfolder = 'results'
    filename = 'imgnet171_mlp_seed_{}.pkl'.format(seed)
    pickle_file_path = os.path.join(master_folder_path, subfolder, filename)
    with open(pickle_file_path, 'rb') as handle:
        metrics = pickle.load(handle)
    
    for metric in ['passes','time']:
            subfolder = 'imagenet171_mlp'
            subsubfolder = 'seed_{}'.format(seed)
            pdf_filename = 'imagenet171_'+'trajs'+'_val_acc_'+metric+'_seed_{}.pdf'.format(seed)
            os.makedirs(os.path.join(master_folder_path, subfolder, subsubfolder), exist_ok=True)
            dir = os.path.join(master_folder_path, subfolder, subsubfolder, pdf_filename)
            multi_joy_plot(metrics, metric, colors, dir)
