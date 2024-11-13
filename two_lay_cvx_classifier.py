# this file is the entrance to the CRONOS package eventually 

from models import CVX_ReLU_MLP
from models import CVX_GReLU_MLP
from optimizers import run
import jax.numpy as jnp
import jax.random as jrn

class Two_Lay_CVX_Classifier:

    def __init__(self, 
                 X_train: jnp.ndarray, 
                 y_train: jnp.ndarray, 
                 P_S: int,
                 model_type = 'CReLU', 
                 beta: float = 10**-3, 
                 seed: int = 0 ) -> None:
        '''
        Model wrapper class for a two-layer convex neural network classifier. Two different model types are supported: \n
        1). CReLU: Convex neural network model with the ReLU activation as described in Mishkin, Sahiner and Pilanci (2022). 
        This is a more computationally efficient version of the convex reformulation introduced in Pilanci and Ergen (2021). \n
        2). GReLU: trains a convex neural network using GReLU activation function as described in Mishkin et al. (2022). \n

        X_train (jnp.ndarray): training data for the model. \n
        y_train (jnp.ndarray): training labels corresponding to X_train. \n
        P_S (int): Number of ReLU/GreLU patterns used to train the model. \n
        model_type (str): Specifies the model type (CReLU/GReLU). Default value is CReLU.\n
        beta (float): L2-regularization parameter in neural network objective. 
        Empirically the final result does not seem to be very sensitive to this parameter. Default value is 1e-3 \n. 
        seed (int): Random seed used to train the model.   
        
        '''
        
        self.coef_ = None
        self.W1 = None
        self.w2 = None
        self.model_type = model_type
        
        # This value of rho is a dummy value for instantiating the model. True value that is used is specified by the user in fit
        rho = 0
        if model_type == 'CReLU':
           self.model = CVX_ReLU_MLP(X_train, y_train, P_S, beta, rho, jrn.key(seed))
        elif model_type == 'GReLU':
           self.model = CVX_GReLU_MLP(X_train, y_train, P_S, beta, rho, jrn.key(seed))
        else:
            raise ValueError(f"We do not support the model type: {model_type}")
        
        self.model.init_model()

    def fit(self, 
            rho: float, 
            max_iters: int, 
            nys_appx_rank: int = 10, 
            pcg_iters: int = 20, 
            gamma_ratio: int = 1,
            Xval: jnp.ndarray = None,
            yval: jnp.ndarray = None, 
            check_opt: bool = False, 
            verbose: bool = True)->None:
        """Fits two-layer convex neural network classifier using the CRONOS algorithm from 
        Feng, Frangella, and Pilanci (2024).\n
        
        rho (float): ADMM penalty parameter. \n
        max_iters (int): How many iterations to run CRONOS for, \n
        pcg_iters (int): How many iterations of PCG to run to solve the subproblem. \n
        X_val (jnp.ndarray): Validation data for monitoring validation loss. \n
        y_val (jnp.ndarray): Validation labels.\n
        check_opt (bool): Flag to determine whether we check (True) or do not check (False) the ADMM primal and dual residuals. \n
        verbose (bool): Flag to determine whether we print optimizer progress. \n

        """
        
        cronos_params = {}
        cronos_params['rank'] = nys_appx_rank
        cronos_params['admm_iters'] = max_iters
        cronos_params['beta'] = self.model.beta
        cronos_params['pcg_iters'] = pcg_iters
        cronos_params['gamma_ratio'] = gamma_ratio
        cronos_params['check_opt'] = check_opt
        cronos_params['verbose'] = verbose

        self.model.rho = rho
        self.model.Xval = Xval
        self.model.yval = yval
        
        print("Beginning model fit")
        coef_, _ = run(self.model, cronos_params, self.model_type)
        self.coef_=coef_
        self.W1, self.w2 = self.model.get_ncvx_weights(self.coef_)
        print("Finished fitting model") 
    
    def predict(self, X: jnp.ndarray)->jnp.ndarray:
        """Returns the predicted class labels for input data X using the model.\n
        X (jnp.ndarray): input data we wish to predict the labels for.
        """   
        if self.coef_ is None:
           raise ValueError("You must fit the model first to make predictions. Use the fit method to do this")
        return self.model.predict(X, self.W1, self.w2)
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray)->float:
        """Computes the percentage accuracy of the model on input data X with true labels y.\n
        X (jnp.ndarray): input data we wish to predict the labels for.\n
        y (jnp.ndarray): true labels corresponding to X.
        """   
        if self.coef_ is None:
            raise ValueError("You must fit the model first to make predictions. Use the fit method to do this") 
        y_hat = jnp.sign(self.predict(X))
        correct_count = jnp.count_nonzero(y==y_hat)
        return correct_count*100/y.shape[0]


