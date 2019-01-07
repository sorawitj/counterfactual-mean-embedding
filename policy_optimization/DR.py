from DirectClassification import *


class DR(DirectClassification):

    def __init__(self, null_feature_vec, null_propensity, null_reward, params=(40, 1024, 200)):
        super().__init__(null_feature_vec, null_reward, params)
        self.null_propensity = null_propensity

    def estimate(self, target_feature_vec, target_propensity=None):
        null_predictions = self.fast_predict.predict(self.null_feature_vec)
        null_predictions = np.array([p['probabilities'][1] for p in null_predictions])

        target_predictions = self.fast_predict.predict(target_feature_vec)
        target_predictions = np.array([p['probabilities'][1] for p in target_predictions])

        ips_weight = target_propensity / self.null_propensity
        estimated_rewards = target_predictions + (self.null_reward - null_predictions) * ips_weight
        return estimated_rewards
