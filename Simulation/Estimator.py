from abc import abstractmethod

import numpy as np
import itertools
import scipy.linalg
import pandas as pd

"""
Classes that represent different policy estimators for simulated experiments
"""

class Estimator(object):
    def __init__(self, n_reco, null_policy, sim_data):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param sim_data: list of observations {"x": context, "y": recommendation, "r": reward, "p": pick item}
        """
        self.n_reco = n_reco
        self.null_policy = null_policy
        self.sim_data = sim_data

    @abstractmethod
    def estimate(self, context, null_reco, null_reward, new_reco):
        """
        calculate expected reward from an observation
        :param context: a string represent user
        :param null_reco: a treatment generated under null policy
        :param null_reward: a reward obtained under null policy
        :param new_reco: a treatment generated under new policy
        :return: expected reward (double)
        """
        pass

class DirectEstimator(Estimator):
    def __init__(self, n_reco, sim_data):
        super(DirectEstimator, self).__init__(n_reco, None, sim_data)
        self.expected_val = self.getExpectedVal()

    def getExpectedVal(self):
        """
        Compute empirical average reward given context, treatment
        :return: a dictionary mapping from (context, treatment) -> average reward
        """
        return_dict = pd.DataFrame(self.sim_data).groupby(['x', 'y'])['r'].mean().to_dict()
        return return_dict

    def estimate(self, context, null_reco, null_reward, new_reco):
        expVal = self.expected_val.get((context, tuple(new_reco)))
        if expVal is not None:
            return expVal
        else:
            return 0.0


class IPSEstimator(Estimator):
    def __init__(self, n_reco, null_policy, new_policy):
        """
        :param new_policy: a policy that we want to estimate its reward
        self.xxx_probDist is a mapping from context to probability of showing a given permutation of items
        """
        super(IPSEstimator, self).__init__(n_reco, null_policy, None)

        self.new_policy = new_policy
        self.null_probDist = dict([(x, self.get_probDist(x, self.null_policy)) for x in self.null_policy.prob])
        self.new_probDist = dict([(x, self.get_probDist(x, self.new_policy)) for x in self.new_policy.prob])

    def get_probDist(self, context, policy):
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

    def estimate(self, context, null_reco, null_reward, new_reco):
        nullProb = self.null_probDist[context][tuple(null_reco)]
        newProb = self.new_probDist[context][tuple(null_reco)]

        return null_reward * newProb / nullProb


class SlateEstimator(IPSEstimator):
    def __init__(self, n_reco, null_policy):
        super(IPSEstimator, self).__init__(n_reco, null_policy, None)
        self.null_probDist = dict([(x, self.get_probDist(x, null_policy)) for x in self.null_policy.prob])

    def gammaInverse(self, context):
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

    def estimate(self, context, null_reco, null_reward, new_reco):
        numAllowedDocs = self.null_policy.n_items
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = validDocs * numAllowedDocs
        tempRange = range(validDocs)

        exploredMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        exploredMatrix[tempRange, null_reco[0:validDocs]] = null_reward

        newMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        newMatrix[tempRange, new_reco[0:validDocs]] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gammaInverse(context), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)


"""
An improved version of Slate estimator
Add one more assumption to reduce variance:
    reward function does not depend on the position of items, for example, showing item i on position 1 is similar to showing item i on position 3
"""
class SlateEstimatorImproved(SlateEstimator):
    def __init__(self, n_reco, null_policy):
        super(SlateEstimatorImproved, self).__init__(n_reco, null_policy)

    def gammaInverse(self, x):

        numAllowedDocs = self.null_policy.n_items

        validDocs = self.n_reco

        gamma = np.zeros((numAllowedDocs, numAllowedDocs),
                         dtype=np.longdouble)
        currentMarginals = np.zeros(numAllowedDocs)
        for p in range(validDocs):
            currentMarginals += np.sum(self.null_probDist[x], axis=tuple([q for q in range(validDocs) if q != p]),
                                       dtype=np.longdouble)
        gamma[0:numAllowedDocs, 0:numAllowedDocs] = np.diag(currentMarginals)

        pairMarginals = np.zeros((numAllowedDocs, numAllowedDocs),
                                 dtype=np.longdouble)
        for p in range(validDocs):
            for q in range(p + 1, validDocs):
                pairMarginals += np.sum(self.null_probDist[x],
                                        axis=tuple([r for r in range(validDocs) if r != p and r != q]),
                                        dtype=np.longdouble)
        np.fill_diagonal(pairMarginals, 0)

        for p in range(numAllowedDocs):
            for q in range(p + 1, numAllowedDocs):
                sum = pairMarginals[p, q] + pairMarginals[q, p]
                gamma[p, q] = sum
                gamma[q, p] = sum

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def estimate(self, context, null_reco, null_reward, new_reco):
        numAllowedDocs = self.null_policy.n_items
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = numAllowedDocs

        exploredMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        exploredMatrix[null_reco[0:validDocs]] = null_reward

        newMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        newMatrix[new_reco[0:validDocs]] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gammaInverse(context), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)

"""
The counterfactual mean embedding estimator 
"""
class CMEstimator(Estimator):

    def __init__(self, n_reco, null_policy, sim_data, context_kernel, recom_kernel, params):
        """
        :param context_kernel: the kernel function for the context variable
        :param recom_kernel: the kernel function for the recommendation
        :param params: all parameters including regularization parameter and kernel parameters
        """
        
        super(CMEstimator,self).__init__(n_reco, null_policy, sim_data)
        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self.params = params

    def estimate(self, context, null_reco, null_reward, new_reco):
        """
        Calculate and return a coefficient vector (beta) of the counterfactual
        mean embedding of reward distribution.
        """

        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        contextMatrix = self.context_kernel(...,..., context_param)
        newContextMatrix = self.context_kernel(...,..., context_param)
        recomMatrix = self.recom_kernel(...,..., recom_param)
        newRecomMatrix = self.recom_kernel(...,..., recom_param)
        
        # calculate the coefficient vector
        b = np.dot(np.multiply(newContextMatrix,newRecomMatrix),np.repeat(1./m,m,axis=0))
        beta_vec = np.linalg.solve(np.multiply(ContextMatrix,RecomMatrix) + np.diag(np.repeat(n*reg_param,n)), b)
        
        # return the expected reward
        return np.dot(beta_vec, null_reward)
        
###
