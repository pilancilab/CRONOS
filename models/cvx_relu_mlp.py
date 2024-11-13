# optimized in JAX for parallelization: updated Nov 1, 2024

from models.cvx_mlp import Convex_MLP
import jax.numpy as jnp
from jax import jit, tree_util
from jax.nn import relu
from jax import jit, tree_util, vmap

class CVX_ReLU_MLP(Convex_MLP):
    def __init__(self, X, y, P_S, beta, rho, seed, d_diags = None, e_diags = None):
        super().__init__(X, y, P_S, beta, rho, seed)
        self.d_diags = d_diags
        self.e_diags = e_diags
    # def __init__(self, X, y, n_classes, P_S, beta, rho, seed,
    # d_diags = None, e_diags = None, Xtst = None, ytst = None):
    #     super().__init__(X, y, P_S, beta, rho, seed)
    #     self.n_classes = n_classes
    #     self.d_diags = d_diags
    #     self.e_diags = e_diags
    #     self.Xtst = Xtst
    #     self.ytst = ytst

    def init_model(self):
        from utils.model_utils import get_hyperplane_cuts

        self.d_diags, self.seed = get_hyperplane_cuts(self.X, self.P_S, self.seed)
        self.e_diags = 2*self.d_diags-1
    
    def matvec_Fi(self, i, vec):
        return self.d_diags[:,i] * (self.X @ vec)
    
    def rmatvec_Fi(self, i, vec):
        return  self.X.T @ (self.d_diags[:,i] * vec)
    
    @jit
    def matvec_F(self, vec):
        #print("****************** matvec_F ******************")
        matvec_Fi_vmap = vmap(self.matvec_Fi, in_axes=(0, 1))
        out2 = jnp.sum(matvec_Fi_vmap(jnp.arange(self.P_S), vec[0] - vec[1]), axis=0)
        return out2
    
    @jit
    def rmatvec_F(self, vec):
        #print("****************** rmatvec_F ******************")
        n, d = self.X.shape
        rmatvec_Fi_vmap = vmap(self.rmatvec_Fi, in_axes=(0, None), out_axes=1)
        out2 = rmatvec_Fi_vmap(jnp.arange(self.P_S), vec)
        out2 = jnp.stack((out2, -out2))
        return out2
    
    def matvec_Gi(self, i, vec):
        return self.e_diags[:,i] * (self.X @ vec)
    
   
    def rmatvec_Gi(self, i, vec):
        return self.X.T@(self.e_diags[:,i]*vec)
    
    @jit
    def matvec_G(self, vec):
        #print("****************** matvec_G ******************")
        n, d = self.X.shape
        matvec_Gi_vmap = vmap(self.matvec_Gi, in_axes=(0, 1), out_axes=1)
        matvec_Gi_vmap_v2 = vmap(matvec_Gi_vmap, in_axes=(None,0))
        out3 = matvec_Gi_vmap_v2(jnp.arange(self.P_S), vec)
        return out3
    
    @jit
    def rmatvec_G(self,vec):
        #print("****************** rmatvec_G ******************")
        n, d = self.X.shape
        rmatvec_Gi_vmap = vmap(self.rmatvec_Gi, in_axes=(0, 1), out_axes=1)
        rmatvec_Gi_vmap_v2 = vmap(rmatvec_Gi_vmap, in_axes=(None, 0))
        out3 = rmatvec_Gi_vmap_v2(jnp.arange(self.P_S), vec)
        return out3

    
    @jit 
    def matvec_A(self, vec):
        b = vec  # test 
        b = b + 1/self.rho * self.rmatvec_F(self.matvec_F(vec))
        # print(b.shape)
        print(self.rmatvec_G(self.matvec_G(vec)).shape)
        b = b + self.rmatvec_G(self.matvec_G(vec))
        return b
    
    def get_ncvx_weights(self, u): #theres a problem here 
        from utils.model_utils import relu_optimal_weights_transform

        return relu_optimal_weights_transform(u[0], u[1])
    
    def predict(self, data, W1, w2):
        return relu(data @ W1) @ w2
    
    def _tree_flatten(self):
        children = (self.X, self.y, self.beta, self.seed, self.d_diags, self.e_diags)  # dynamic values
        aux_data = {'P_S': self.P_S, 'rho': self.rho}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        X, y, beta, seed, d_diags, e_diags = children
        P_S = aux_data['P_S']
        rho = aux_data['rho']
        return cls(X,y,P_S,beta,rho,seed,d_diags,e_diags)
  
tree_util.register_pytree_node(CVX_ReLU_MLP,
                               CVX_ReLU_MLP._tree_flatten,
                               CVX_ReLU_MLP._tree_unflatten)