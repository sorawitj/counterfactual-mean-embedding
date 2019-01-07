import numpy as np

class wIPS(object):

    def __init__(self, null_propensity, null_reward):
        self.null_propensity = null_propensity
        self.null_reward = null_reward
        
    def estimate(self, target_propensity):
        """
        Calculate and return a vector of the weighted inverse propensity scores.
        """
        
        ips_weight = target_propensity / self.null_propensity
        exp_reward = ips_weight * self.null_reward
        exp_weight = np.mean(ips_weight)

        return exp_reward / exp_weight
        
