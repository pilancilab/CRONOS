# import specific functions or classes for convenience
import os 

from .train import train
# from .other_module import AnotherClass
# from .two_lay_cvx_classifier import Two_Lay_CVX_Classifier

# Set package-level metadata
__version__ = '0.1.0'
__author__ = 'miria'


# initialize package-level variables and logging if needed
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")