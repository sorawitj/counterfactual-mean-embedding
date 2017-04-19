import numpy as np
from collections import defaultdict
from itertools import groupby


class PriorPolicy(object):
    def __init__(self, prob, n_hotels, n_reco, greedy=True):
        self.prob = prob
        self.n_reco = n_reco
        self.n_hotels = n_hotels
        self.greedy = greedy

    def recommend(self, x):
        cur_dist = self.prob[x]
        if self.greedy:
            reco = np.argsort(-cur_dist, kind='mergesort')[:self.n_reco]
        else:
            reco = np.random.choice(len(cur_dist), self.n_reco, p=cur_dist, replace=False)
        return reco


class GlobalSortPolicy(object):
    def __init__(self, n_hotels, n_reco):
        self.global_sort = None
        self.n_reco = n_reco
        self.n_hotels = n_hotels

    def recommend(self, x):
        return self.global_sort[:self.n_reco]

    def train(self, simData):
        book_hotel = [d['h'] for d in simData if d['r'] > 0]
        hotel_booking = defaultdict(int)
        for h in book_hotel:
            hotel_booking[h] += 1
        self.global_sort = np.array(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1])))


class MostCommonByUserPolicy(object):
    def __init__(self, n_hotels, n_reco):
        self.policy_map = dict()
        self.n_reco = n_reco
        self.n_hotels = n_hotels

    def recommend(self, x):
        return self.policy_map[x][:self.n_reco]

    def train(self, simData):
        groupData = groupby(sorted(simData, key=lambda x: x['x']), key=lambda x: x['x'])
        for x, data in groupData:
            book_hotel = [d['h'] for d in data if d['r'] > 0]
            hotel_booking = defaultdict(int)
            for h in book_hotel:
                hotel_booking[h] += 1
            self.policy_map[x] = np.array(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1])))


class RandomSortPolicy(object):
    def __init__(self, n_hotels, n_reco):
        self.global_sort = None
        self.n_reco = n_reco
        self.n_hotels = n_hotels

    def recommend(self, x):
        return np.random.choice(self.n_hotels, self.n_reco, replace=False)

class FixedPolicy(object):
    def __init__(self, fixed_reco):
        self.fixed_reco = fixed_reco

    def recommend(self, x):
        return self.fixed_reco[x]
