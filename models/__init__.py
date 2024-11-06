# import specific functions or classes for convenience
import os 

from .cnn import main_function, helper_function
from .cvx_grelu_mlp import AnotherClass
from .cvx_mlp import Convex_MLP
from .cvx_relu_mlp import CVX_ReLU_MLP
from .get_model import init_model
from .grelu_mlp import GReLU_MLP
from .relu_mlp import ReLU_MLP
from .two_layer_mlp import Two_Layer_ReLU_MLP

# Set package-level metadata
__version__ = '0.1.0'
__author__ = 'miria'


# # initialize package-level variables and logging if needed
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")