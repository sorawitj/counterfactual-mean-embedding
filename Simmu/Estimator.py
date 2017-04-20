import numpy as np
import itertools
import scipy.linalg
import pandas as pd


class DirectEstimator(object):
    def __init__(self, n_reco, new_policy, simData):
        self.simData = simData
        self.new_policy = new_policy
        self.n_reco = n_reco
        self.new_probDist = dict([(x, self.get_probDist(x, self.new_policy)) for x in self.new_policy.prob])
        self.expectedVal = self.getExpectedVal()

    def get_probDist(self, x, policy):
        numAllowedDocs = policy.n_hotels
        currentDistribution = policy.prob[x]
        validDocs = self.n_reco

        slates = np.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                          dtype=np.float32)
        for x in itertools.permutations(range(numAllowedDocs), validDocs):
            currentDenom = currentDistribution.sum(dtype=np.longdouble)
            slateProb = 1.0
            for p in range(validDocs):
                slateProb *= (currentDistribution[x[p]] / currentDenom)
                currentDenom -= currentDistribution[x[p]]
                if currentDenom <= 0:
                    break
            slates[tuple(x)] = slateProb
        return slates

    def getExpectedVal(self):
        return pd.DataFrame(self.simData).groupby(['x', 'y'])['r'].mean().to_dict()

    def estimate(self, x, explored_ranking, explored_value, new_ranking):
        expVal = self.expectedVal.get((x, tuple(new_ranking)))
        if expVal is not None:
            return expVal
        else:
            return 0.0


class IPSEstimator(object):
    def __init__(self, n_reco, null_policy, new_policy):
        self.null_policy = null_policy
        self.new_policy = new_policy
        self.n_reco = n_reco

        self.null_probDist = dict([(x, self.get_probDist(x, self.null_policy)) for x in self.null_policy.prob])
        self.new_probDist = dict([(x, self.get_probDist(x, self.new_policy)) for x in self.new_policy.prob])

    def get_probDist(self, x, policy):
        numAllowedDocs = policy.n_hotels
        currentDistribution = policy.prob[x]
        validDocs = self.n_reco

        slates = np.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                          dtype=np.float32)
        for x in itertools.permutations(range(numAllowedDocs), validDocs):
            currentDenom = currentDistribution.sum(dtype=np.longdouble)
            slateProb = 1.0
            for p in range(validDocs):
                slateProb *= (currentDistribution[x[p]] / currentDenom)
                currentDenom -= currentDistribution[x[p]]
                if currentDenom <= 0:
                    break
            slates[tuple(x)] = slateProb
        return slates

    def estimate(self, x, explored_ranking, explored_value, new_ranking):
        nullProb = self.null_probDist[x][tuple(explored_ranking)]
        newProb = self.new_probDist[x][tuple(explored_ranking)]

        return explored_value * newProb / nullProb


class SlateEstimator(object):
    def __init__(self, n_reco, policy):
        self.policy = policy
        self.n_reco = n_reco

        self.slates = dict([(x, self.get_slate(x)) for x in self.policy.prob])

    def get_slate(self, x):
        numAllowedDocs = self.policy.n_hotels
        currentDistribution = self.policy.prob[x]
        validDocs = self.n_reco

        slates = np.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                          dtype=np.float32)
        for x in itertools.permutations(range(numAllowedDocs), validDocs):
            currentDenom = currentDistribution.sum(dtype=np.longdouble)
            slateProb = 1.0
            for p in range(validDocs):
                slateProb *= (currentDistribution[x[p]] / currentDenom)
                currentDenom -= currentDistribution[x[p]]
                if currentDenom <= 0:
                    break
            slates[tuple(x)] = slateProb
        return slates

    def gammaInverse(self, x):

        numAllowedDocs = self.policy.n_hotels

        validDocs = self.n_reco

        gamma = np.zeros((numAllowedDocs * validDocs, numAllowedDocs * validDocs),
                         dtype=np.longdouble)
        for p in range(validDocs):
            currentStart = p * numAllowedDocs
            currentEnd = p * numAllowedDocs + numAllowedDocs
            currentMarginals = np.sum(self.slates[x], axis=tuple([q for q in range(validDocs) if q != p]),
                                      dtype=np.longdouble)
            gamma[currentStart:currentEnd, currentStart:currentEnd] = np.diag(currentMarginals)

        for p in range(validDocs):
            for q in range(p + 1, validDocs):
                currentRowStart = p * numAllowedDocs
                currentRowEnd = p * numAllowedDocs + numAllowedDocs
                currentColumnStart = q * numAllowedDocs
                currentColumnEnd = q * numAllowedDocs + numAllowedDocs
                pairMarginals = np.sum(self.slates[x],
                                       axis=tuple([r for r in range(validDocs) if r != p and r != q]),
                                       dtype=np.longdouble)
                np.fill_diagonal(pairMarginals, 0)

                gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd] = pairMarginals
                gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd] = pairMarginals.T

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def estimate(self, x, explored_ranking, explored_value, new_ranking):
        numAllowedDocs = self.policy.n_hotels
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = validDocs * numAllowedDocs
        tempRange = range(validDocs)

        exploredMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        exploredMatrix[tempRange, explored_ranking[0:validDocs]] = explored_value

        newMatrix = np.zeros((validDocs, numAllowedDocs), dtype=np.longdouble)
        newMatrix[tempRange, new_ranking[0:validDocs]] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gammaInverse(x), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)


class SlateEstimator2(object):
    def __init__(self, n_reco, policy):
        self.policy = policy
        self.n_reco = n_reco

        self.slates = dict([(x, self.get_slate(x)) for x in self.policy.prob])

    def get_slate(self, x):
        numAllowedDocs = self.policy.n_hotels
        currentDistribution = self.policy.prob[x]
        validDocs = self.n_reco

        slates = np.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                          dtype=np.float32)
        for x in itertools.permutations(range(numAllowedDocs), validDocs):
            currentDenom = currentDistribution.sum(dtype=np.longdouble)
            slateProb = 1.0
            for p in range(validDocs):
                slateProb *= (currentDistribution[x[p]] / currentDenom)
                currentDenom -= currentDistribution[x[p]]
                if currentDenom <= 0:
                    break
            slates[tuple(x)] = slateProb
        return slates

    def gammaInverse(self, x):

        numAllowedDocs = self.policy.n_hotels

        validDocs = self.n_reco

        gamma = np.zeros((numAllowedDocs, numAllowedDocs),
                         dtype=np.longdouble)
        currentMarginals = np.zeros(numAllowedDocs)
        for p in range(validDocs):
            currentMarginals += np.sum(self.slates[x], axis=tuple([q for q in range(validDocs) if q != p]),
                                       dtype=np.longdouble)
        gamma[0:numAllowedDocs, 0:numAllowedDocs] = np.diag(currentMarginals)

        pairMarginals = np.zeros((numAllowedDocs, numAllowedDocs),
                                 dtype=np.longdouble)
        for p in range(validDocs):
            for q in range(p + 1, validDocs):
                pairMarginals += np.sum(self.slates[x],
                                        axis=tuple([r for r in range(validDocs) if r != p and r != q]),
                                        dtype=np.longdouble)
        np.fill_diagonal(pairMarginals, 0)

        for p in range(numAllowedDocs):
            for q in range(p + 1, numAllowedDocs):
                sum = pairMarginals[p, q] + pairMarginals[q, p]
                gamma[p, q] = sum
                gamma[q, p] = sum

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def estimate(self, x, explored_ranking, explored_value, new_ranking):
        numAllowedDocs = self.policy.n_hotels
        validDocs = min(numAllowedDocs, self.n_reco)
        vectorDimension = numAllowedDocs

        exploredMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        exploredMatrix[explored_ranking[0:validDocs]] = explored_value

        newMatrix = np.zeros(numAllowedDocs, dtype=np.longdouble)
        newMatrix[new_ranking[0:validDocs]] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = np.dot(self.gammaInverse(x), posRelVector)

        return np.dot(estimatedPhi, newSlateVector)
