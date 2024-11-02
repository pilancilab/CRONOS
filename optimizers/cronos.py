import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.nn import relu
from optimizers.pcg import pcg
from preconditioner.nystrom import Nys_Precond, rand_nys_appx
from utils.metric_utils import mse, compute_bin_acc
from utils.proximal_utils import proxl2_tensor
from time import perf_counter
from typing import NamedTuple, Tuple


class CRONOS_State(NamedTuple):
    u: jnp.ndarray
    v: jnp.ndarray
    s: jnp.ndarray
    lam: jnp.ndarray
    nu: jnp.ndarray
    Gu: jnp.ndarray

# Helper to initialize CRONOS state
def init_cronos_state(d, n, P_S):
    u = jnp.zeros((2, d, P_S))
    v = jnp.zeros((2, d, P_S))
    s = jnp.zeros((2, n, P_S))
    lam = jnp.zeros((2, d, P_S))
    nu = jnp.zeros((2, n, P_S))
    return CRONOS_State(u=u, v=v, s=s, lam=lam, nu=nu, Gu=jnp.zeros((2, n, P_S)))

# Helper for validation metrics
def compute_validation_metrics(metrics, u, model):
    y_hat = model.matvec_F(u)
    W1, w2 = model.get_ncvx_weights(u)
    y_hat_val = model.predict(model.Xval, W1, w2)

    metrics['train_loss'].append(mse(y_hat, model.y))
    metrics['val_loss'].append(mse(y_hat_val, model.yval))
    metrics['train_acc'].append(compute_bin_acc(y_hat, model.y))
    metrics['val_acc'].append(compute_bin_acc(y_hat_val, model.yval))

    return metrics

# Function that executes 1 step of CRONOS
@partial(jit, static_argnames=['pcg_iters'])
def cronos_step(state: CRONOS_State, 
                model,
                Mnys, 
                beta: float, 
                gamma_ratio: float, 
                pcg_iters: int,
                pcg_tol: float):
    
    b_1 = model.rmatvec_F(model.y) / model.rho
    b = b_1 + state.v - state.lam + model.rmatvec_G(state.s - state.nu)

    # u update via PCG
    u, _, _ = pcg(b, model, Mnys, pcg_iters, pcg_tol)

    # v update using prox operator
    v = state.v.at[0].set(proxl2_tensor(u[0] + state.lam[0], beta=beta, gamma=1 / model.rho))
    v = v.at[1].set(proxl2_tensor(u[1] + state.lam[1], beta=beta, gamma=1 / model.rho))

    # s update using ReLU
    Gu = model.matvec_G(u)
    s = relu(Gu + state.nu)

    # dual updates
    lam = state.lam + (u - v) * gamma_ratio
    nu = state.nu + (Gu - s) * gamma_ratio

    return CRONOS_State(u=u, v=v, s=s, lam=lam, nu=nu, Gu=Gu)

@partial(jit, static_argnames=['beta'])
def opt_conds(state: CRONOS_State, model, beta: float)->Tuple[float,float,float]:
    y_hat = model.matvec_F(state.u)
    u_v_dist = jnp.linalg.norm(state.u - state.v) + jnp.linalg.norm(state.Gu - state.s)
    u_optimality = jnp.linalg.norm(model.rmatvec_F(y_hat - model.y.squeeze()) + model.rho * (state.lam + model.rmatvec_G(state.nu)))
    v_optimality = jnp.linalg.norm(beta * state.v / jnp.linalg.norm(state.v, axis=2, keepdims=True) - model.rho * state.lam)
    return u_v_dist, u_optimality, v_optimality

# Function that runs CRONOS optimizer
def run(model, admm_params: dict, model_type: str):
    # Extract params for CRONOS
    rank = admm_params['rank']
    beta = admm_params['beta']
    gamma_ratio = admm_params['gamma_ratio']
    admm_iters = admm_params['admm_iters']
    pcg_iters = admm_params['pcg_iters']
    check_opt = admm_params['check_opt']
    verbose = admm_params['verbose']

    validate = model.Xval is not None
    n, d = model.X.shape

    # Initialize state
    state = init_cronos_state(d, n, model.P_S)

    # Initialize metrics
    if validate:
        metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'times': []}

        metrics = compute_validation_metrics(metrics, state.u, model)

        # Intialize best model dict:
        best_model_dict = {'v': state.u[0], 'w': state.u[1], 'iteration': 0}   

    # Nystrom approximation and preconditioner
    U, S, model.seed = rand_nys_appx(model, rank, model_type, model.seed)
    Mnys = Nys_Precond(U, S, d, model.rho, model.P_S, model_type)

    for k in range(admm_iters):
        start = perf_counter()
        
        # Perform CRONOS step
        state = cronos_step(state, model, Mnys, beta, gamma_ratio, pcg_iters, 1/(1+k)**1.2)

        if check_opt:
            u_v_dist, u_optimality, v_optimality = opt_conds(state, model, beta)
            if verbose == True:
                print(f"iter: {k}\n  u-v dist = {u_v_dist}, u resid = {u_optimality}, v resid = {v_optimality}\n")
        
        t_iter = perf_counter() - start
        metrics['times'].append(t_iter)

        if validate == True:
            metrics = compute_validation_metrics(metrics, state.u, model)
            
            if metrics['val_acc'][k]>metrics['val_acc'][k-1]:
                best_model_dict['v'] = state.u[0]
                best_model_dict['w'] = state.u[1]
                best_model_dict['iteration'] = k
            
            elif metrics['val_acc'][k]==metrics['val_acc'][k-1] and metrics['val_loss'][k]<metrics['val_loss'][k-1]:
                best_model_dict['v'] = state.u[0]
                best_model_dict['w'] = state.u[1]
                best_model_dict['iteration'] = k
            
            if verbose == True:
               print(f"Iteration: {k}")
               print(f"Train Loss: {metrics['train_loss'][-1]}")
               print(f"Train Accuracy: {metrics['train_acc'][-1]}")
               print(f"Validation Loss: {metrics['val_loss'][-1]}")
               print(f"Validation Accuracy: {metrics['val_acc'][-1]}\n")
            
            # Early stopping
            if k>=10 and k % 10 == 0 and metrics['val_acc'][k]<=metrics['val_acc'][best_model_dict['iteration']]:
                print("Validation accuracy is flat or decreasing. CRONOS will now terminate and return the best model found")
                return (best_model_dict['v'], best_model_dict['w']), metrics    

    return (state.u[0], state.u[1]), metrics


