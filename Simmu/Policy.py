from abc import abstractmethod

import numpy as np
from collections import defaultdict
from itertools import groupby

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
class ProbPolicy(Policy):


    def __init__(self, prob_dist, n_items, n_reco, greedy=True):
        """
        :param prob_dist: probability distribution over items
        :param greedy: if greedy is true -> recommend items which have the highest probabilities
        """
        super(ProbPolicy, self).__init__(n_items, n_reco)
        self.prob = prob_dist
        self.greedy = greedy

    def recommend(self, context):
        cur_dist = self.prob[context]
        if self.greedy:
            reco = np.argsort(-cur_dist, kind='mergesort')[:self.n_reco]
        else:
            reco = np.random.choice(len(cur_dist), self.n_reco, p=cur_dist, replace=False)
        return reco


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
        groupData = groupby(sorted(sim_data, key=lambda x: x['x']), key=lambda x: x['x'])
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
