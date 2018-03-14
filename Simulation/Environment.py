import numpy as np
from scipy.special import expit
from Utils import *

"""
Classes represent environments which define how rewards are generated
"""


class Environment(object):
    def __init__(self, item_vectors, context_dim, examine_rate=None):
        r"""
        initialize simple environment

        :param item_vectors: a dictionary mapping from user(context) to their preferences(probability distribution over items)
        :param examine_rate:
        """
        self.item_vectors = item_vectors
        self.examine_rate = examine_rate
        self.context_dim = context_dim

    def get_context(self):
        return np.random.normal(size=(self.context_dim,))

    def get_reward(self, context_features, reco):
        r"""
        generate a reward given user(context) and recommendation
        :param context: a string represent user
        :param reco: a permutation of item
        :return: 1 if the pick item is in the recommendation "and" user examine the pick item else 0
        """
        click_probs = softmax(np.matmul(context_features, self.item_vectors[reco].T))
        click = np.random.choice(np.arange(len(click_probs)), p=click_probs)
        if self.examine_rate is None:
            examine = len(reco)
        else:
            examine = np.random.geometric(self.examine_rate, 1)
        # non_zero = click.nonzero()[0]
        # if non_zero.size > 0:
        #     reward = 1.0 / (non_zero[0] + 1)
        # else:
        #     reward = 0.0
        reward = 1.0 / (click + 1)
        return reward


class AvgEnvironment(object):
    def __init__(self, item_vectors, context_dim):
        r"""
        initialize simple environment

        :param context_vectors: a dictionary mapping from user(context) to their preferences(probability distribution over items)
        :param examine_rate:
        """
        self.item_vectors = item_vectors
        self.context_dim = context_dim

    def get_context(self):
        return np.random.normal(0, 1.0, size=(self.context_dim,))

    def get_reward(self, context_features, reco):
        r"""
        generate a reward given user(context) and recommendation
        :param context: a string represent user
        :param reco: an avg vector of recommended items
        :return: 1 if the pick item is in the recommendation "and" user examine the pick item else 0
        """
        reco_vector = np.mean(self.item_vectors[reco], axis=0)
        prob = expit(context_features.dot(reco_vector))
        reward = np.random.binomial(1, prob)
        return reward


class NNEnvironment(AvgEnvironment):
    def __init__(self, item_vectors, context_dim):
        super().__init__(item_vectors, context_dim)

    def get_reward(self, context_features, reco):
        reco_vector = np.mean(self.item_vectors[reco], axis=0)
        all_vector = np.concatenate([context_features, reco_vector])
        W1 = np.random.normal(size=(all_vector.shape[0], 100))
        B1 = np.random.normal(size=(100,))
        W2 = np.random.normal(size=(100, 1))
        B2 = np.random.normal(size=(1,))
        hidden1 = expit(all_vector.dot(W1) + B1)
        prob = expit((hidden1.dot(W2) + B2) + np.random.normal(0, 1.0))
        reward = np.random.binomial(1, prob)[0]
        return reward


class BinaryDiversEnvironment(object):
    """
    more complicated environment in which ADA does not hold true (Reward function depends on the interaction of items in the recommendation)
    """

    def __init__(self, examine_rate, book_rate, p, hotels_per_group):
        self.examine_rate = examine_rate
        self.book_rate = book_rate
        self.p = p
        self.hotels_per_group = hotels_per_group

    def get_reward(self, x, reco):
        if self.examine_rate >= 1:
            examine = self.examine_rate
        else:
            examine = min(np.random.geometric(self.examine_rate), len(reco))

        interest = np.random.choice(2, p=self.p)

        groups = reco[:examine] // self.hotels_per_group
        matches = (groups == interest) & (np.random.rand(examine) < self.book_rate)
        if matches.any():
            pick = matches.argmax()
            reward = 1.0
        else:
            pick = None
            reward = 0.0

        return reward, pick
