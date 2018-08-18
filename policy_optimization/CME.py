from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class CME(object):

    def __init__(self, null_feature_vec, null_rewards):
        reg_pow = -1
        self.reg_param = (10.0 ** reg_pow) / null_feature_vec.shape[0]
        self.kernel_param = (10.0 ** 0)
        self.null_feature_vec = null_feature_vec
        self.null_rewards = null_rewards

        # use median heuristic for the bandwidth parameters
        null_kernel_param = (0.5 * self.kernel_param) / np.median(pdist(null_feature_vec, 'sqeuclidean'))

        nullContextMatrix = rbf_kernel(null_feature_vec, null_feature_vec, null_kernel_param)

        self.n = null_feature_vec.shape[0]
        A = nullContextMatrix + np.diag(np.repeat(self.n * self.reg_param, self.n))
        self.A_inv = scipy.linalg.inv(A)

    def estimate(self, target_feature_vec):
        """
         Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
         """

        target_kernel_param = (0.5 * self.kernel_param) / np.median(pdist(target_feature_vec, 'sqeuclidean'))

        targetContextMatrix = rbf_kernel(self.null_feature_vec, target_feature_vec, target_kernel_param)

        B = np.dot(targetContextMatrix, np.repeat(1.0 / self.n, self.n, axis=0))

        beta_vec = np.matmul(self.A_inv, B)

        target_reward = beta_vec * self.null_rewards

        return target_reward
