# import specific functions or classes for convenience
import os 

#from .core import main_function, helper_function
#from .other_module import AnotherClass
from .two_lay_cvx_classifier import Two_Lay_CVX_Classifier
from .config import *

# Set package-level metadata
__version__ = '0.1.0'
__author__ = 'miria'


# # initialize package-level variables and logging if needed
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")

# OUTPUT_DIR = os.path.join(PACKAGE_DIR, 'results')

# # create the results directory if it doesn't exist
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)
#     print(f"Created directory: {OUTPUT_DIR}")
# else:
#     print(f"Directory already exists: {OUTPUT_DIR}")
