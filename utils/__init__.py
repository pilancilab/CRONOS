# import specific functions or classes for convenience
import os 

from .gpt2_dataloader import load_data
from .linops_utils import tensor_to_vec, vec_to_tensor
from .load_data import load_cifar, load_food, load_imagenet, load_imagenet512
from .metric_utils import get_model_performance, mse, compute_bin_acc, classification_accuracy
from .model_utils import get_grelu_patterns, grelu_optimal_weights_transform, optimal_weights_transform
from .opt_utils import get_optimizer
from .proximal_utils import proxl2_vec, proxl2_tensor, batch_proxl2_tensor
from .train_utils import get_batch
#from .typing_utils import


# Set package-level metadata
__version__ = '0.1.0'
__author__ = 'miria'


# # initialize package-level variables and logging if needed
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")