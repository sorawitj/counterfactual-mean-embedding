import numpy as np

"""
Classes represent environments which define how rewards are generated
"""

class Environment(object):
    def __init__(self, probDist, examine_rate=None):
        r"""
        initialize simple environment

        :param probDist: a dictionary mapping from user(context) to their preferences(probability distribution over items)
        :param examine_rate:
        """
        self.prob = probDist
        self.examine_rate = examine_rate

    def get_reward(self, context, reco):
        r"""
        generate a reward given user(context) and recommendation
        :param context: a string represent user
        :param reco: a permutation of item
        :return: 1 if the pick item is in the recommendation "and" user examine the pick item else 0
        """
        cur_dist = self.prob[context]
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




