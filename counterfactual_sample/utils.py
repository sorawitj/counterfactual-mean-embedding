import numpy as np
from scipy.spatial.distance import cdist

from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def get_mixture_gaussian_samples(n, d, locs, weights):
    p = weights/weights.sum()
    samples = []
    for i in range(n):  # iteratively draw samples
        Z = np.random.choice(np.arange(len(p)), p=p)  # latent variable
        samples.append(np.random.multivariate_normal(mean=locs[Z], cov=np.eye(d), size=1))
    return np.array(samples).squeeze()


def gen_data(ns):
    # generate data from the marginal distributions P(X_0) and P(X_1)
    d = 5
    X0 = np.random.randn(ns, d)
    X1 = get_mixture_gaussian_samples(ns, d, np.array([[-5, 2.5, 0, 0, 2.5],
                                                       [2.5, 2.5, 0, 0, -5],
                                                       [2.5, -5, 0, 0, 2.5]]), np.array([1, 1, 1]))
    # generate Y_0 and Y_1 from the conditional models
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Y0 = np.dot(beta_vec, X0.T) + 0.1 * np.random.randn(X0.shape[0])  # potential outcome Y_0
    Y1 = np.dot(beta_vec, X1.T) + 0.1 * np.random.randn(X1.shape[0])  # potential outcome Y_1

    return X0, X1, Y0, Y1


# kernel function
def gauss_rbf(X1, X2, sigma=1):
    """
    The Gaussian RBF kernel function.
    X1 : the first nxd input data matrix. Each row corresponds to data point
    X2 : the second mxd input data matrix. Each row corresponds to the data point
    """

    K = np.exp(-np.divide(cdist(X1, X2, 'sqeuclidean'), 2 * sigma))

    return K


def find_best_params(X1, Y1, reg_grid=[1e1, 1e0, 0.1, 1e-2],
                     gamma_grid=[1e1, 1e0, 0.1, 1e-2], num_cv=5):
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=num_cv,
                      param_grid={"alpha": reg_grid,
                                  "gamma": gamma_grid})
    kr.fit(X1, Y1)

    sg = 1.0 / kr.best_params_['gamma']
    reg_param = kr.best_params_['alpha']

    return sg, reg_param
