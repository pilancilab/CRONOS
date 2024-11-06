import os
import pickle
import numpy as np

def get_best_config(folder_path):
    # List to store the loaded data along with filenames
    #configs = {}
    #loaded_data_with_filenames = []
    best_val = 0
    best_config = None

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a pickle file by extension
        if os.path.isfile(file_path) and filename.endswith('.pkl'):
            # Load the pickle file
            with open(file_path, 'rb') as file:
                results = pickle.load(file, encoding='latin1')
                # Append a tuple of filename and data to the list
                config_best_val = np.max(results['CRONOS']['val_acc'])
                if config_best_val>best_val:
                  best_val = config_best_val
                  best_config = filename

    return best_config, best_val
