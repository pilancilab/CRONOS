import os
import pickle
from gpt2_plots import plot_tuned
from get_config import get_best_config

#### GPT2-medium-seqhead ####
master_folder_path = '/Users/zacharyfrangella/Documents/Python/Cronos Plotting/pickle_results/PARAM_SEARCH_fri/gpt2seqhead_med'

best_config, _ = get_best_config(master_folder_path)
print(best_config)
pickle_file_path = os.path.join(master_folder_path, best_config)
with open(pickle_file_path, 'rb') as handle:
        results = pickle.load(handle)

plot_path = '/Users/zacharyfrangella/Documents/Python/Cronos Plotting/plots'
subfolder = 'gpt2seqhead_med'
directory =  os.makedirs(os.path.join(master_folder_path, subfolder), exist_ok=True)

plot_tuned(results, directory)




