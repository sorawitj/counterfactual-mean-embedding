from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class wIPS(object):

    def __init__(self, n_reco: int, null_policy: MultinomialPolicy, target_policy: MultinomialPolicy):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param target_policy: a policy that we want to estimate its reward
        """
        self.n_reco = n_reco
        self.null_policy = null_policy
        self.target_policy = target_policy

    def calculate_weight(self, row):
        nullProb = self.null_policy.get_propensity(row.null_multinomial, row.null_reco)
        if not self.target_policy.greedy:
            targetProb = self.target_policy.get_propensity(row.target_multinomial, row.null_reco)
        else:
            targetProb = 1.0 if row.null_reco == row.target_reco else 0

        return targetProb / nullProb
        
    def estimate(self, data):
        """
        Calculate and return a vector of the weighted inverse propensity scores.
        """

        data['ips_w'] = data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(data['ips_w'] * data['null_reward'])
        exp_weight = np.mean(data['ips_w'])

        return exp_reward / exp_weight
        
