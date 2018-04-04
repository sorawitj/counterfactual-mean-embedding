from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import joblib


def simulate_data(null_policy, target_policy, environment, item_vectors, seed):
    """
    simulate data given policy, environment and set of context
    :return: observations
    """
    np.random.seed(seed)
    context_vec = environment.get_context()
    null_reco, null_multinomial = null_policy.recommend(context_vec)
    # recommendation is represented by a concatenation of recommended item vectors
    null_reco_vec = np.concatenate(item_vectors[null_reco])
    null_reward = environment.get_reward(context_vec, null_reco)

    target_reco, target_multinomial = target_policy.recommend(context_vec)
    # recommendation is represented by a concatenation of recommended item vectors
    target_reco_vec = np.concatenate(item_vectors[target_reco])
    target_reward = environment.get_reward(context_vec, target_reco)

    observation = {"context_vec": context_vec, "null_reco": tuple(null_reco),
                   "null_reco_vec": null_reco_vec, "null_reward": null_reward,
                   "target_reco": tuple(target_reco), "null_multinomial": null_multinomial,
                   "target_multinomial": target_multinomial,
                   "target_reco_vec": target_reco_vec, "target_reward": target_reward}
    return observation


def grid_search(params, estimator, sim_data, n_iterations):
    actual_value = sim_data.target_reward.mean()
    return_df = pd.DataFrame(columns=['param', 'estimated_value', 'actual_value', 'error'])
    for param in params:
        estimated_values = []
        for _ in range(n_iterations):
            estimator.params = param
            estimated_values.append(estimator.estimate(sim_data))
        mean_value = np.array(estimated_values).mean()
        ret = {'param': param, 'estimated_value': mean_value, 'actual_value': actual_value,
               'error': np.abs(mean_value - actual_value),
               'percent_error': 100.0 * np.abs(mean_value - actual_value) / actual_value}
        return_df = return_df.append(ret, ignore_index=True)

    return return_df


def compare_estimators(estimators, null_policy, target_policy, environment, item_vectors,
                       config, n_iterations=1):
    return_df = pd.DataFrame(columns=[e.name for e in estimators] + ['actual_value'])
    for i in range(n_iterations):
        seeds = np.random.randint(np.iinfo(np.int32).max, size=config['n_observation'])
        responses = joblib.Parallel(n_jobs=-2)(
            joblib.delayed(simulate_data)(null_policy, target_policy, environment, item_vectors, seeds[i]) for i in
            range(config['n_observation'])
        )

        sim_data = pd.DataFrame(responses)

        actual_value = sim_data.target_reward.mean()
        estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
        estimated_values['actual_value'] = actual_value
        estimated_values['null_reward'] = sim_data.null_reward.mean()
        for e in estimators:
            estimated_values[e.name + '_square_error'] = (estimated_values[e.name] - actual_value) ** 2
        print(estimated_values)
        return_df = return_df.append(estimated_values, ignore_index=True)

    return return_df


if __name__ == "__main__":
    config = {
        "n_items": 40,
        "n_reco": 5,
        "n_observation": 5000,
        "context_dim": 10
    }
    result_df = dict()

    target_item_vectors = np.random.normal(0, 1, size=(config['n_items'], config['context_dim']))
    for multiplier in [0.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]:
        null_item_vectors = target_item_vectors - multiplier * target_item_vectors

        # The policy we use to generate sim data
        null_policy = MultinomialPolicy(null_item_vectors, config['n_items'], config['n_reco'], temperature=0.1)

        # The target policy
        target_policy = MultinomialPolicy(target_item_vectors, config['n_items'], config['n_reco'], temperature=0.5)

        env_item_vectors = 0.5 * target_item_vectors
        environment = AvgEnvironment(env_item_vectors, config['context_dim'])

        reg_pow = np.arange(-1, 0)
        reg_params = (10.0 ** reg_pow) / config['n_observation']
        bw_params = 10.0 ** -1
        params = [reg_params, bw_params, bw_params]
        """
         Comparing between estimators
         """
        estimators = [IPSEstimator(config['n_reco'], null_policy, target_policy),
                      CMEstimator(rbf_kernel, rbf_kernel, params), DirectEstimator()]

        compare_df = compare_estimators(estimators, null_policy, target_policy, environment, env_item_vectors, config,
                                        5)
        result_df[multiplier] = compare_df
