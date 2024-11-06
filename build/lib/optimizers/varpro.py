import jax.numpy as jnp
import optax
from jax import jit, tree_util
from optax.contrib import dadapt_adamw

class VarPro:
    def __init__(self, 
                 lr: float, 
                 weight_decay: float) -> object:
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer =  optax.multi_transform({'last_layer': optax.set_to_zero(),
  'outer_layers': dadapt_adamw(learning_rate = lr, weight_decay = weight_decay)},
  {'params': {'last_layer': 'last_layer', 'outer_layers': 'outer_layers'}})
    
    @jit
    def step(self,
         grads: dict,
         params: dict,
         opt_state,
         model: object,
         Xtr: jnp.ndarray,
         ytr: jnp.ndarray):
        
        #Get last layer weights 
        Ztr = model.apply(params, Xtr, method = model.apply_outer_layers)
        shift = 10**-8
        w = jnp.linalg.solve(Ztr.T@Ztr+shift*jnp.eye(Ztr.shape[1]), Ztr.T@ytr)
        w = w.reshape(w.shape[0],1)
        params['params']['last_layer']['Dense_0']['kernel'] = w
  
        #Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'lr': self.lr, 'weight_decay': self.weight_decay}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        () = children
        lr = aux_data['lr']
        rho = aux_data['weight_decay']
        return cls(lr, rho)

tree_util.register_pytree_node(VarPro,
                                VarPro._tree_flatten,
                                VarPro._tree_unflatten)