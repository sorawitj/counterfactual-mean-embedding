import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""On develop"""

def to2d(theta, r):
    return (np.array([np.cos(theta), np.sin(theta)]) * r).T


class MFWorld(object):
    def __init__(self,
            n_hotel=100,
            n_user=100,
            model_error=1.0,
            examine_rate=0.2,
            booking_threshold=0.7,
            n_reco=10,
            seed = None
    ):
        np.random.seed(seed)
        self.n_user = n_user
        self.n_hotel = n_hotel
        self.booking_threshold = booking_threshold
        self.n_reco = n_reco

        self.hotels = to2d(
            np.random.uniform(-np.pi, np.pi, n_hotel),
            1.0 + np.random.randn(n_hotel) * 0.1
        )

        user_angle = np.random.uniform(-np.pi, np.pi, self.n_user)
        user_obs_factors = to2d(user_angle, 1.0)

        real_user_angle = user_angle + np.random.normal(0.0, model_error, self.n_user)
        self.user_length = np.random.exponential(0.5, n_user)
        self.user_real_factors = to2d(real_user_angle, self.user_length)

        self.user_hotel_scores = self.user_real_factors.dot(self.hotels.T)
        self.model_scores = user_obs_factors.dot(self.hotels.T)
        self.hotel_similarity = np.maximum(
            self.hotels.dot(self.hotels.T), 0.0
        )

        if examine_rate < 1.0:
            self.user_hotels_seen = np.minimum(
                np.random.geometric(examine_rate, self.n_user), self.n_reco
            )
        else:
            self.user_hotels_seen = np.ones(n_user, "int32") * examine_rate

    def evaluate_rank(self, rank):
        scores = self.user_hotel_scores[
            np.arange(self.user_hotel_scores.shape[0]).reshape(-1, 1), rank]
        user_hotel_candidates = scores > self.booking_threshold
        first_relevant = np.argmax(
            np.c_[np.zeros(self.n_user, dtype=bool), user_hotel_candidates],
            axis=1
        )
        booking = (
            (first_relevant <= self.user_hotels_seen.reshape(-1)) &
            (0 < first_relevant)
        )
        return booking.mean()

    def get_diversified_rank(self, gamma):
        alread_ranked = np.zeros((self.n_user, 0), "int32")

        model_scores_a = self.model_scores.copy()
        for _ in range(self.n_reco):
            similarity = (self.hotel_similarity[alread_ranked]).sum(axis=1) ** 0.5
            model_scores_r = (1.0 - gamma) * model_scores_a - gamma * similarity
            next_item_idx = np.argmax(model_scores_r, axis=1)
            model_scores_a[np.arange(self.n_user), next_item_idx] = -np.inf
            alread_ranked = np.c_[alread_ranked, next_item_idx]

        return alread_ranked

    def get_model_rank(self):
        return np.argsort(-self.model_scores, axis=1)[:, :self.n_reco]

    def get_random_rank(self, seed=0):
        np.random.seed(seed)
        return np.argsort(np.random.randn(self.n_user, self.n_hotel), axis=1)[:, :self.n_reco]


mf_world = MFWorld(n_user=10000, n_hotel=200, model_error=1.0, examine_rate=5, booking_threshold=0.7)
ranks = [mf_world.get_diversified_rank(0.4), mf_world.get_model_rank(), mf_world.get_random_rank(None)]
for r in ranks:
    print(mf_world.evaluate_rank(r))

# from scipy.misc import comb,factorial
# s = 3
# comb(30, 8) * factorial(8)
