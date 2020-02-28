import functools

from scipy.optimize import minimize

from utils import *

def estimate_cme(X0, X1, Y1):
    sg, reg_param = find_best_params(X1, Y1)
    print(reg_param)
    print(sg)

    K1 = gauss_rbf(X0, X0, sg)
    K2 = gauss_rbf(X0, X1, sg)

    # beta vector
    b = np.dot(np.dot(np.linalg.inv(K1 + reg_param * np.eye(X0.shape[0])), K2),
               np.ones((X1.shape[0], 1)) / X1.shape[0])

    # normalize b for stability
    b = b / b.sum()

    return b

# simplify kernel function to avoid dimension crashing
def rbf_kernel(X1, X2, sigma=1):
    K = np.exp(-((X1 - X2) ** 2) / (2 * sigma))
    return K


# minimization problem in each step
def obj_fun_herding(y, Y0, b, sigma, yt_hist=None, t=None):
    first_term = np.dot(b.T, rbf_kernel(Y0, y, sigma))
    if yt_hist is None:
        return -first_term
    else:
        second_term = rbf_kernel(yt_hist, y, sigma).mean()
    return -first_term + second_term


# the MMD objective
def mmd_obj(ys, yt, bs, sigma):
    mvec = np.repeat(1.0 / len(yt), len(yt))
    first_term = np.dot(np.dot(mvec.T, gauss_rbf(yt, yt, sigma)), mvec)
    second_term = np.dot(np.dot(bs.T, gauss_rbf(ys, yt, sigma)), mvec)
    third_term = np.dot(np.dot(bs.T, gauss_rbf(ys, ys, sigma)), bs)

    return first_term - 2 * second_term + third_term


def generate_herding_samples(num_herding, Y0, sigma, weights):
    # first iteration
    obj_fn = functools.partial(obj_fun_herding, Y0=Y0, b=weights, sigma=sigma, yt_hist=None, t=None)
    #     y0 = Y0.mean(axis=0)
    y0 = np.random.randn()
    res = minimize(obj_fn, y0, method='CG', options={'gtol': 1e-10, 'disp': False})
    yt = res.x.ravel()[0]

    # initialize samples
    yt_samples = [yt]
    obj_val = [mmd_obj(Y0, np.array(yt_samples)[:, np.newaxis], weights, sigma).flatten()]

    # start the iterations 2 to num_herding
    max_trials = 10
    for t in range(2, num_herding + 1):
        yt_hist = np.array(yt_samples)
        obj_fn = functools.partial(obj_fun_herding, Y0=Y0, b=weights, sigma=sigma, yt_hist=yt_hist, t=t)
        res = minimize(obj_fn, y0, method='CG', options={'gtol': 1e-10, 'disp': False})

        num_trials = 0
        while (not res.success and num_trials < max_trials):
            res = minimize(obj_fn, y0 + 1e-10 * np.random.randn(), method='CG', options={'gtol': 1e-10, 'disp': False})
            num_trials += 1

        yt = res.x.ravel()[0]
        yt_samples += [yt]
        obj_val += [mmd_obj(Y0, np.array(yt_samples)[:, np.newaxis], weights, sigma).flatten()]

    return [yt_samples, obj_val]
