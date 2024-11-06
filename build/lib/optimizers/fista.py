import jax 
import jax.numpy as jnp
from jax import jit
from optimizers.pcg import pcg
from preconditioner.nystrom import Nys_Precond, rand_nys_appx
from utils.metric_utils import mse, compute_bin_acc
from utils.model_utils import optimal_weights_transform
from utils.proximal_utils import proxl2_tensor
from functools import partial

def fista(model, u0, max_iter):
  eta, t = 10**-2, 1
  y = jnp.copy(u0)
  t_old = jnp.copy(t)
  n = model.X.shape[0]
  for i in range(max_iter):
    grad = model.rmatvec_F((model.matvec_F(u0)-model.y)/n)
    u = proxl2_tensor(u0-eta*grad, eta*model.beta, 1)
    t = (1+jnp.sqrt(1+4*t_old**2))/2
    y = u+((t_old-1)/t)*(u-u0)
    t_old = jnp.copy(t)
    u0 = jnp.copy(u)
  return y
  
