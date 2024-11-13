import jax.numpy as jnp
import optax
from jax import jit, tree_util
from optax.contrib import dadapt_adamw
from models import CVX_ReLU_MLP
from optimizers import admm

from optimizers.dist_shampoo.distributed_shampoo import distributed_shampoo

class CronosAM:
    def __init__(self,
                 cronos_params: dict, 
                 lr: float, 
                 weight_decay: float) -> object:
        self.cronos_params = cronos_params
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer =  optax.multi_transform({'head': optax.set_to_zero(),
   'features': dadapt_adamw(learning_rate = lr, weight_decay = weight_decay)},
   {'params': {'head': 'head', 'features': 'features'}})

    def get_last_two_layers(self, model, params, Xtr, ytr):
       """ Method that computes last two layers for input NN model by solving the convex reformulation with Cronos. """
       
       # Pass data matrix through layers 1:L-2
       tildX = model.apply(params, Xtr, method = model.features_transform)
       
       cvx_head = CVX_ReLU_MLP(tildX, ytr, self. cronos_params['P_S'], self.cronos_params['beta'], self.cronos_params['rho'], 
                                self.cronos_params['seed'])
      
       cvx_head.init_model()
       
       P_S =  self.cronos_params['P_S']
       from utils import optimal_weights_transform
       # Solves convex reformulation for the last two layers via ADMM
       cvx_weights, _ = admm(cvx_head, self.cronos_params['admm_params'])

       penult_layer, last_layer = optimal_weights_transform(cvx_weights[0], cvx_weights[1], P_S, tildX.shape[1])
       params['params']['head']['Dense_0']['kernel'] = penult_layer
       params['params']['head']['Dense_1']['kernel'] = last_layer.reshape(2*P_S, 1)
      
       return params

    @jit 
    def outer_layers_step(self,
                grads: dict, 
                params: dict,
                opt_state):
                """Method that takes one step on the feature layers of the network"""
  
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

tree_util.register_pytree_node(CronosAM,
                                CronosAM._tree_flatten,
                                CronosAM._tree_unflatten)