import jax.numpy as jnp
from models.two_layer_mlp import Two_Layer_ReLU_MLP


def init_model(model_params, x, key):
    if model_params['type'] == 'two_layer_mlp':
        model = Two_Layer_ReLU_MLP() 
        params = model.init(key, x)
    else:
      raise ValueError("This model is currently not implemented.")
    
    def loss(params, data_batch, data_labels):
        preds = model.apply(params, data_batch)
        return jnp.sum(((preds-data_labels)**2).mean())
    
    return params, model, loss

