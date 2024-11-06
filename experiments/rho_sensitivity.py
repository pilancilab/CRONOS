import jax
import jax.numpy as jnp
import jax.random as jrn
from models.cvx_relu_mlp import CVX_ReLU_MLP
from optimizers.admm import admm
from utils.load_data import load_cifar, load_fmnist



training_X, training_y, test_X, test_y = load_fmnist()

n_classes = 10
P_S = 32
beta = 10**-3
seed = jrn.key(0)


rho_grid = [1, 0.1, 0.01, 0.001]

for rho in rho_grid:
    model = CVX_ReLU_MLP(training_X, training_y, n_classes, P_S, beta, rho, seed)
    model.init_model()
    model.Xtst = test_X
    model.ytst = test_y

    admm_params = dict(rank = 10, beta = beta, gamma_ratio = 1, admm_iters = 5, pcg_iters = 20, check_opt = False)

    cvx_weights, metrics,_ = admm(model, admm_params)

    print(f"Peak val for {rho}: {jnp.max(metrics['val_acc'])}")