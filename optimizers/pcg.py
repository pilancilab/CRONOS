import jax.numpy as jnp
from jax import jit
from jax.lax import while_loop
from functools import partial
from typing import NamedTuple

class PCG_State(NamedTuple):
    u: jnp.ndarray
    r: jnp.ndarray
    z: jnp.ndarray
    p: jnp.ndarray
    r_dot_z: float
    k: int

def pcg(b: jnp.ndarray,
        model: object,
        M: object,
        max_iter: int,
        tol: float = 10**-6):
    
    return _pcg(b, model, M ,max_iter, tol)


@partial(jit, static_argnames=['max_iter'])
def _pcg(b: jnp.ndarray,
        model,
        M,
        max_iter: int,
        tol: float):
    
    def _init_pcg():
      r = b
      z = M.apply(r)
      p = jnp.copy(z)
      r_dot_z = jnp.sum(r*z)
      k = 0
      return PCG_State(jnp.zeros_like(b),r,z,p,r_dot_z,k)
    
    def _pcg_step(state):
      w = model.matvec_A(state.p)
      
      # Update solution and residual
      alpha = state.r_dot_z / jnp.sum(w * state.p)
      u = state.u+alpha * state.p
      r = state.r-alpha * w
      
      # Apply preconditioner
      z = M.apply(r)

      # Update search direction
      rnp1_dot_znp1 = jnp.sum(r * z)
      p = z + (rnp1_dot_znp1 / state.r_dot_z) * state.p
      return PCG_State(u = u,r= r,z = z, p = p,r_dot_z = rnp1_dot_znp1,k =state.k+1)
    
    state = _init_pcg()

    def _cond(state):
        # Stop when residual norm is less than tolerance or max iterations reached
        return (jnp.linalg.norm(state.r) > tol) & (state.k < max_iter)
  
    final_state = while_loop(_cond, _pcg_step, state)

    return final_state.u, final_state.r, final_state.k




