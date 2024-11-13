import jax 
import jax.numpy as jnp
from jax import jit
from utils import mse, compute_bin_acc
# from utils.model_utils import optimal_weights_transform
from functools import partial

def gd_admm(model, admm_params):
    rank = admm_params['rank']
    beta = admm_params['beta']
    gamma_ratio = admm_params['gamma_ratio']
    admm_iters = admm_params['admm_iters']
    pcg_iters = admm_params['pcg_iters']
    check_opt = admm_params['check_opt']

    validate = False
    if model.Xtst is not None:
        validate = True
    
    n, d = model.X.shape

    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
               }

        # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP 
    u = jnp.zeros((2, d, model.P_S))
    # v contrains v1 ... vP, w1 ... wP
    v = jnp.zeros((2, d, model.P_S))
    # slacks s1 ... sP, t1 ... tP
    s = jnp.zeros((2, n, model.P_S))
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = jnp.zeros((2, d, model.P_S))
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = jnp.zeros((2, n, model.P_S))
    
    for _ in range(admm_iters):

        u, v, s, lam, nu, Gu = _admm_step(model, u, v, s, lam, nu)

    return (v[0], v[1]), metrics, model.seed

  
@jit 
def _admm_step (model, u, v, s, lam, nu):
  # u update
  from utils import proxl2_tensor

  grad_loss = model.rmatvec_F(model.matvec_F(u)-model.y)
  u = u-(10**-8)*(grad_loss+model.rho*(u-v+lam+model.rmatvec_G(model.matvec_G(u)-s+nu)))



  # updates on v = (v1...vP, w1...wP) via prox operator
  v = v.at[0].set(proxl2_tensor(u[0] + lam[0], beta = model.beta, gamma=1 / model.rho))
  # w update
  v = v.at[1].set(proxl2_tensor(u[1] + lam[1], beta= model.beta, gamma=1 / model.rho))

  # updates on s = (s1...sP, t1...tP)
  Gu = model.matvec_G(u)
  s = jax.nn.relu(Gu + nu)

  # dual updates on lam=(lam11...lam2P), nu=(nu11...nu2P)
  lam += (u - v)
  nu += (Gu - s)

  return u, v, s, lam, nu, Gu   
