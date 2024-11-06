import jax.numpy as jnp
import optax
from jax import jit, tree_util
from optax.contrib import dadapt_adamw
from optimizers.dist_shampoo.distributed_shampoo import distributed_shampoo
from models.cvx_grelu_mlp import CVX_GReLU_MLP
from optimizers.pcg import pcg
from preconditioner.nystrom import Nys_Precond, rand_nys_appx
from utils.model_utils import grelu_optimal_weights_transform

class CronosPro:
    def __init__(self,
                 cronos_params: dict, 
                 lr: float, 
                 weight_decay: float) -> object:
        self.cronos_params = cronos_params
        self.lr = lr
        self.weight_decay = weight_decay
        #self.optimizer =  optax.multi_transform({'head': optax.set_to_zero(),
  #'features': distributed_shampoo(learning_rate=lr, block_size=128)},
  #{'params': {'head': 'head', 'features': 'head'}})
        self.optimizer =  optax.multi_transform({'head': optax.set_to_zero(),
   'features': dadapt_adamw(learning_rate = lr, weight_decay = weight_decay)},
   {'params': {'head': 'head', 'features': 'features'}})


    def get_last_two_layers(self, model, params, Xtr, ytr):
       """ Method that computes last two layers for input NN model by solving the convex reformulation with Cronos. """
       
       # Pass data matrix through layers 1:L-2
       tildX = model.apply(params, Xtr, method = model.apply_outer_layers)
       
       cvx_head = CVX_GReLU_MLP(tildX, ytr, self.cronos_params['P_S'], self.cronos_params['beta'], self.cronos_params['rho'], 
                                self.cronos_params['seed'])
      
       cvx_head.init_model()
       
       P_S =  self.cronos_params['P_S']

       U, S, cvx_head.seed = rand_nys_appx(cvx_head, self.cronos_params['rank'], 'CGReLU', cvx_head.seed)

       Mnys = Nys_Precond(U, S, tildX.shape[1], cvx_head.rho, P_S, 'CGReLU')

       # Solves convex reformulation for the last two layers via ADMM
       cvx_weights, _ = pcg(cvx_head, Mnys, self.cronos_params['pcg_iters'])

       penult_layer, last_layer = grelu_optimal_weights_transform(cvx_weights, P_S, tildX.shape[1])
       params['params']['head']['Dense_0']['kernel'] = penult_layer
       params['params']['head']['Dense_1']['kernel'] = last_layer.reshape(P_S, 1)
      
       return params, cvx_weights, cvx_head.matvec_F

    @jit 
    def outer_layers_step(self,
                grads: dict, 
                params: dict,
                opt_state):
                """Method that takes one optimizer step on the feature layers of the network"""
  
                # Update parameters
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
  
                return params, opt_state
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'cronos_params': self.cronos_params, 'lr': self.lr, 'weight_decay': self.weight_decay}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        () = children
        cronos_params = aux_data['cronos_params']
        lr = aux_data['lr']
        rho = aux_data['weight_decay']
        return cls(cronos_params, lr, rho)

tree_util.register_pytree_node(CronosPro,
                                CronosPro._tree_flatten,
                                CronosPro._tree_unflatten)