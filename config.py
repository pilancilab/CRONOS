'''
User config file holds all front end parameters
'''

import os
import sys

# initialize package-level variables and logging if needed
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")

OUTPUT_DIR = os.path.join(PACKAGE_DIR, 'results')

# create the results directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")
else:
    print(f"Directory already exists: {OUTPUT_DIR}")


BATCH_SIZE = 100

# this is used in MLP_runner__.py
DATASET = 'imagenet512' # change to 'food', 'imagenet', 'imagenet512'
MODEL = 'mlp' # change to 'mlp', 'cnn', 'gpt2'
