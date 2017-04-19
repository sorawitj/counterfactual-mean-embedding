import numpy as np


class Environment(object):
    def __init__(self, prob, examine_rate=None):
        self.prob = prob
        self.examine_rate = examine_rate

    def get_reward(self, x, reco):
        cur_dist = self.prob[x]
        pick = np.random.choice(len(cur_dist), 1, p=cur_dist, replace=False)[0]
        if self.examine_rate is None:
            examine = len(reco)
        else:
            examine = np.random.geometric(self.examine_rate, 1)
        reward = reco[:examine].__contains__(pick)
        if not reward:
            pick = None
        return int(reward), pick


class BinaryDiversEnvironment(object):
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




