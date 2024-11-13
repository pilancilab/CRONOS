# import specific functions or classes for convenience
import os 

# from .adamW_nojit import main_function, helper_function
from .adamW import AdamW
from .admm import admm
from .cronos_am import CronosAM
from .cronos_pro import CronosPro
from .cronos import compute_validation_metrics, cronos_step, run
from .dadapt_adamW import Dadapt_AdamW
from .fista import fista
from .gd_admm import gd_admm
#from .pcg import pcg, _pcg
from .sgd import SGD
from .shampoo import Shampoo
from .varpro import VarPro
from .yogi import Yogi

# Set package-level metadata
__version__ = '0.1.0'
__author__ = 'miria'


# # initialize package-level variables and logging if needed
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(f"CRONOS package loaded from directory: {PACKAGE_DIR}")