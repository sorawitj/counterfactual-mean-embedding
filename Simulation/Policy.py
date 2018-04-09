from abc import abstractmethod

import numpy as np
from collections import defaultdict
import itertools
from Utils import *

"""
Classes represent policies that have recommend function mapping from context to recommendation(treatment)
"""


class Policy(object):
    def __init__(self, n_items, n_reco):
        """
        :param n_items: number of all items
        :param n_reco: number of recommendation
        """
        self.n_reco = n_reco
        self.n_items = n_items

    @abstractmethod
    def recommend(self, context):
        """
        recommend a permutation of items given a context
        :return: list of items
        """
        pass


"""
Sample items without replacement based on pre-define probability
"""


class MultinomialPolicy(Policy):
    def __init__(self, item_vectors, estimated_vector_vectors, n_items, n_reco, temperature=1.0, greedy=False):
        """
        :param item_vectors: probability distribution over items
        :param greedy: if greedy is true -> recommend items which have the highest probabilities
        """
        super(MultinomialPolicy, self).__init__(n_items, n_reco)
        self.item_vectors = item_vectors
        self.estimated_vector_vectors = estimated_vector_vectors
        self.greedy = greedy
        self.tau = temperature

    def get_propensity(self, multinomial, reco):
        """
        Calculate probability of given recommendation set
        """
        prob = 1.0
        current_denom = multinomial.sum()
        for p in range(self.n_reco):
            prob *= (multinomial[reco[p]] / current_denom)
            current_denom -= multinomial[reco[p]]
            if current_denom <= 0:
                break

        return prob

    def recommend(self, user):
        user_vector = self.estimated_vector_vectors[user, :]
        multinomial = softmax(np.matmul(user_vector, self.item_vectors.T), tau=self.tau)

        if self.greedy:
            reco = np.argsort(-multinomial, kind='mergesort')[:self.n_reco]
        else:
            reco = np.random.choice(len(multinomial), self.n_reco, p=multinomial, replace=False)
        return reco, multinomial, user_vector


"""
Sort items by popularity (number of clicks)
"""


class GlobalSortPolicy(Policy):
    def __init__(self, n_items, n_reco, sim_data):
        super(GlobalSortPolicy, self).__init__(n_items, n_reco)
        self.global_sort = self.get_mostpopular(sim_data)

    def get_mostpopular(self, sim_data):
        book_hotel = [d['h'] for d in sim_data if d['r'] > 0]
        hotel_booking = defaultdict(int)
        for h in book_hotel:
            hotel_booking[h] += 1
        return np.array(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1])))

    def recommend(self, context):
        return self.global_sort[:self.n_reco]


"""
Sort items by popularity given context(user)
"""


class MostCommonByUserPolicy(Policy):
    def __init__(self, n_items, n_reco, sim_data):
        super(MostCommonByUserPolicy, self).__init__(n_items, n_reco)
        self.sorting_map = self.get_mostpopular(sim_data)
        self.n_reco = n_reco
        self.n_items = n_items

    def get_mostpopular(self, sim_data):
        groupData = itertools.groupby(sorted(sim_data, key=lambda x: x['x']), key=lambda x: x['x'])
        for x, data in groupData:
            book_hotel = [d['h'] for d in data if d['r'] > 0]
            hotel_booking = defaultdict(int)
            for h in book_hotel:
                hotel_booking[h] += 1
        return np.array(list(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1]))))

    def recommend(self, context):
        return self.sorting_map[context][:self.n_reco]


"""
Random sort
"""


class RandomSortPolicy(Policy):
    def __init__(self, n_items, n_reco):
        super(RandomSortPolicy, self).__init__(n_items, n_reco)
        self.n_reco = n_reco
        self.n_items = n_items

    def recommend(self, context):
        return np.random.choice(self.n_items, self.n_reco, replace=False)


class FixedPolicy(object):
    def __init__(self, fixed_reco):
        self.fixed_reco = fixed_reco

    def recommend(self, x):
        return self.fixed_reco[x]
