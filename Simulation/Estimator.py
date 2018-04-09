from abc import abstractmethod

import numpy as np
import itertools
import scipy.linalg
import pandas as pd

from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import pdist

"""
Classes that represent different policy estimators for simulated experiments
"""


class Estimator(object):
    @abstractmethod
    def estimate(self, sim_data):
        """
        calculate expected reward from an observation
        :param sim_data: a data frame consists of {context, null_reco, null_reward, new_reco, new_reward}
        :return: expected reward (double)
        """
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class DirectEstimator(Estimator):
    @property
    def name(self):
        return "direct_estimator"

    def estimate(self, sim_data):
        expReward = sim_data.groupby(['context', 'new_reco'])['null_reward'].agg(['mean', 'count'])
        expReward = (expReward['mean'] * expReward['count']).sum()

        return expReward


class IPSEstimator(Estimator):
    @property
    def name(self):
        return "ips_estimator"

    def __init__(self, n_reco, null_policy, new_policy):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param new_policy: a policy that we want to estimate its reward

        self.xxx_probDist is a mapping from context to probability of showing a given permutation of items
        """
        self.n_reco = n_reco
        self.null_policy = null_policy
        self.new_policy = new_policy
        self.null_prob_dist = dict([(x, self.get_prob_dist(x, self.null_policy)) for x in self.null_policy.prob])
        self.new_prob_dist = dict([(x, self.get_prob_dist(x, self.new_policy)) for x in self.new_policy.prob])

    def get_prob_dist(self, context, policy):
        """
        Calculate probability distribution over set of permutation of items
        :param context: a string represent user
        :param policy: any policy
        :return: an array mapping from permutation of item -> probability, e.x. probDist[(1, 2, 3)]: 0.002
        """
        numAllowedDocs = policy.n_items
        currentDistribution = policy.prob[context]
        validDocs = self.n_reco

        probDist = np.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                            dtype=np.float32)
        for permutation in itertools.permutations(range(numAllowedDocs), validDocs):
            currentDenom = currentDistribution.sum(dtype=np.longdouble)
            prob = 1.0
            for p in range(validDocs):
                prob *= (currentDistribution[permutation[p]] / currentDenom)
                currentDenom -= currentDistribution[permutation[p]]
                if currentDenom <= 0:
                    break
            probDist[tuple(permutation)] = prob
        return probDist

    def single_estimate(self, context, null_reco, null_reward, new_reco):
        nullProb = self.null_prob_dist[context][tuple(null_reco)]
        newProb = self.new_prob_dist[context][tuple(null_reco)]

        return null_reward * newProb / nullProb

    def estimate(self, sim_data):
        expReward = sim_data.apply(
            lambda x: self.single_estimate(x.context, x.null_reco, x.null_reward, x.new_reco), axis=1).sum()

        return expReward


class SlateEstimator(IPSEstimator):
    @property
    def name(self):
        return "slate_estimator"

    def __init__(self, n_reco, null_policy):

        self.n_reco = n_reco
        self.null_policy = null_policy
        self.null_prob_dist = dict([(x, self.get_prob_dist(x, null_policy)) for x in self.null_policy.prob])

    def gamma_inverse(self, context):
        """
        calculate inverse of gamma
        see Off-policy evaluation for slate recommendation(https://arxiv.org/pdf/1605.04812.pdf) for more details
        :return: gamma matrix inverse
        """

        numAllowedDocs = self.null_policy.n_items

        validDocs = self.n_reco

        gamma = np.zeros((numAllowedDocs * validDocs, numAllowedDocs * validDocs),
                         dtype=np.longdouble)
        for p in range(validDocs):
            currentStart = p * numAllowedDocs
            currentEnd = p * numAllowedDocs + numAllowedDocs
            currentMarginals = np.sum(self.prob_dist[context], axis=tuple([q for q in range(validDocs) if q != p]),
                                      dtype=np.longdouble)
            gamma[currentStart:currentEnd, currentStart:currentEnd] = np.diag(currentMarginals)

        for p in range(validDocs):
            for q in range(p + 1, validDocs):
                currentRowStart = p * numAllowedDocs
                currentRowEnd = p * numAllowedDocs + numAllowedDocs
                currentColumnStart = q * numAllowedDocs
                currentColumnEnd = q * numAllowedDocs + numAllowedDocs
                pairMarginals = np.sum(self.prob_dist[context],
                                       axis=tuple([r for r in range(validDocs) if r != p and r != q]),
                                       dtype=np.longdouble)
                np.fill_diagonal(pairMarginals, 0)

                gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd] = pairMarginals
                gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd] = pairMarginals.T

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def single_estimate(self, context, null_reco, null_reward, new_reco):
        numAllowedDocs = self.null_policy.n_items
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = validDocs * numAllowedDocs
        tempRange = range(validDocs)

        exploredMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        exploredMatrix[tempRange, list(null_reco[0:validDocs])] = null_reward

        newMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        newMatrix[tempRange, list(new_reco[0:validDocs])] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gamma_inverse(context), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)


"""
An improved version of Slate estimator
Add one more assumption to reduce variance:
    reward function does not depend on the position of items, for example, showing item i on position 1 is similar to showing item i on position 3
"""


class SlateEstimatorImproved(SlateEstimator):
    def __init__(self, n_reco, null_policy):
        super(SlateEstimatorImproved, self).__init__(n_reco, null_policy)

    def gamma_inverse(self, x):

        numAllowedDocs = self.null_policy.n_items

        validDocs = self.n_reco

        gamma = np.zeros((numAllowedDocs, numAllowedDocs),
                         dtype=np.longdouble)
        currentMarginals = np.zeros(numAllowedDocs)
        for p in range(validDocs):
            currentMarginals += np.sum(self.null_prob_dist[x], axis=tuple([q for q in range(validDocs) if q != p]),
                                       dtype=np.longdouble)
        gamma[0:numAllowedDocs, 0:numAllowedDocs] = np.diag(currentMarginals)

        pairMarginals = np.zeros((numAllowedDocs, numAllowedDocs),
                                 dtype=np.longdouble)
        for p in range(validDocs):
            for q in range(p + 1, validDocs):
                pairMarginals += np.sum(self.null_prob_dist[x],
                                        axis=tuple([r for r in range(validDocs) if r != p and r != q]),
                                        dtype=np.longdouble)
        np.fill_diagonal(pairMarginals, 0)

        for p in range(numAllowedDocs):
            for q in range(p + 1, numAllowedDocs):
                sum = pairMarginals[p, q] + pairMarginals[q, p]
                gamma[p, q] = sum
                gamma[q, p] = sum

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def single_estimate(self, context, null_reco, null_reward, new_reco):
        numAllowedDocs = self.null_policy.n_items
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = numAllowedDocs

        exploredMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        exploredMatrix[list(null_reco[0:validDocs])] = null_reward

        newMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        newMatrix[list(new_reco[0:validDocs])] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gamma_inverse(context), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)


"""
The counterfactual mean embedding estimator 
"""


class CMEstimator(Estimator):
<<<<<<< HEAD
    
    def __init__(self, context_kernel, recom_kernel, params=(1e-5,1.0,1.0)):
=======
    @property
    def name(self):
        return "cme_estimator"

    def __init__(self, context_kernel, recom_kernel, params):
>>>>>>> c1ff08e642866b37525d6b31cafa180ba39c4b78
        """
         :param context_kernel: the kernel function for the context variable
         :param recom_kernel: the kernel function for the recommendation
         :param params: all parameters including regularization parameter and kernel parameters
         """

        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self.params = params

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, value):
        self.__params = value

    def estimate(self, sim_data):
        """
         Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
         """

        # extract the regularization and kernel parameters
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]
        
        null_reward = sim_data.null_reward
        context_vec = np.stack(sim_data.context_vec.as_matrix())
        null_reco_vec = np.stack(sim_data.null_reco_vec.as_matrix())
        new_reco_vec = np.stack(sim_data.new_reco_vec.as_matrix())

        # use median heuristic for the bandwidth parameters
        context_param      = 0.5/np.median(pdist(context_vec,'seuclidean'))
        null_recom_param   = (0.5*context_param)/np.median(pdist(null_reco_vec,'seuclidean'))
        new_recom_param    = (0.5*recom_param)/np.median(pdist(new_reco_vec,'seuclidean'))
        
        contextMatrix = self.context_kernel(context_vec, context_vec, context_param)
        newContextMatrix = self.context_kernel(context_vec, context_vec, context_param)
        recomMatrix = self.recom_kernel(null_reco_vec, null_reco_vec, null_recom_param)
        newRecomMatrix = self.recom_kernel(new_reco_vec, new_reco_vec, new_recom_param)

        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["new_reco"].shape[0]
        n = sim_data["null_reco"].shape[0]
        b = np.dot(np.multiply(newContextMatrix, newRecomMatrix), np.repeat(1. / m, m, axis=0))

        # solve a linear least-square
        beta_vec = np.linalg.solve(np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n)), b)
        #beta_vec[beta_vec < 0] = 0.0

        # solve a linear least-squares problem with bounds on the weight vector
        #beta_vec = lsq_linear(np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n)), b, bounds=(0.,np.inf), lsmr_tol='auto', max_iter=40, verbose=2).x

        # use non-negative least square solver
        #beta_vec, res = nnls(np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n)), b)
        
        # return the expected reward as an average of the rewards, obtained from the null policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        return np.dot(beta_vec, null_reward)
