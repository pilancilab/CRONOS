import numpy as np
import jax.numpy as jnp
import jax.random as jrn

# import model class
from two_lay_cvx_classifier import Two_Lay_CVX_Classifier

X = np.random.randn(25, 5)
w = np.random.randn(5)/np.sqrt(5)
y = np.sign(X@w)

X = jnp.array(X, dtype=jnp.float32)
w = jnp.array(w, dtype=jnp.float32)
y = jnp.array(y, dtype=jnp.float32)

# train - validation split
X_train = X[0:16,:]
y_train = y[0:16]
X_val = X[16:20,:]
y_val = y[16:20]
X_tst = X[21:25,:]
y_tst = y[21:25]



# Number of neurons to use
P_S = 40
#Type of network
model_type = 'CReLU'
# L2-regularization parameter 
beta = 10**-3
# Random seed
seed = jrn.key(0)

# Define classification model
clf = Two_Lay_CVX_Classifier(X, y, 40, 'CReLU', beta=10**-3, seed=0)

# ADMM dampling parameter
rho = 0.1
# Number of ADMM iters
max_iter = 30
# Preconditioner rank
nys_appx_rank = 10
# PCG iters
pcg_iters = 20

# Fit the model
clf.fit(rho, max_iter, nys_appx_rank, pcg_iters, Xval=X_val, yval=y_val, check_opt=False, verbose=True)

# Predict test set labels
y_pre = clf.predict(X_tst)

# Compute classification accuracy on test set
print("Computing test accuracy.")
test_acc = clf.score(X_tst, y_tst)
print(f"Test set accuracy: {test_acc}")

