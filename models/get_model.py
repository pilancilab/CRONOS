import jax
import jax.numpy as jnp

def init_model(model_params, x, key):
    if model_params['type'] == 'relu-mlp':
        from models import ReLU_MLP
        model = ReLU_MLP()
        params = model.init(key, x)
    elif model_params['type'] == 'two_layer_mlp':
        from models import Two_Layer_ReLU_MLP

        model = Two_Layer_ReLU_MLP() 
        params = model.init(key, x)
    elif model_params['type'] == 'varpro-mlp':
        from models import VarPro_MLP
        model = VarPro_MLP()
        params = model.init(key, x)
    elif model_params['type'] == 'cnn':
        from models import CNN

        model = CNN()
        params = model.init(key, jnp.ones((1,x.shape[0],x.shape[1],x.shape[2])))
    elif model_params['type'] == 'varpro-cnn':
        from models import VarPro_CNN

        model = VarPro_CNN()
        params = model.init(key, jnp.ones((1,x.shape[0],x.shape[1],x.shape[2])))
    else:
      raise ValueError("This model is currently not implemented.")
    
    def loss(params, data_batch, data_labels):
        preds = model.apply(params, data_batch)
        return ((preds-data_labels)**2).mean()
    
    return params, model, loss
