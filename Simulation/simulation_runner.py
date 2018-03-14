from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import joblib


def simulate_data(null_policy, new_policy, environment, item_vectors, seed):
    """
    simulate data given policy, environment and set of context
    :return: observations
    """
    np.random.seed(seed)
    context_vec = environment.get_context()
    null_reco, null_multinomial = null_policy.recommend(context_vec)
    # recommendation is represented by an average over recommended item vectors
    # null_reco_vec = np.mean(item_vectors[null_reco], axis=0)
    null_reco_vec = np.concatenate(item_vectors[null_reco])
    null_reward = environment.get_reward(context_vec, null_reco)

    new_reco, new_multinomial = new_policy.recommend(context_vec)
    # recommendation is represented by an average over recommended item vectors
    # new_reco_vec = np.mean(item_vectors[new_reco], axis=0)
    new_reco_vec = np.concatenate(item_vectors[new_reco])
    new_reward = environment.get_reward(context_vec, new_reco)

    observation = {"context_vec": context_vec, "null_reco": tuple(null_reco),
                   "null_reco_vec": null_reco_vec, "null_reward": null_reward, "new_reco": tuple(new_reco),
                   "null_multinomial": null_multinomial, "new_multinomial": new_multinomial,
                   "new_reco_vec": new_reco_vec, "new_reward": new_reward}
    return observation


def grid_search(params, estimator, sim_data, n_iterations):
    actual_value = sim_data.new_reward.mean()
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


def compare_estimators(estimators, null_policy, new_policy, environment, item_vectors,
                       config, n_iterations=1):
    return_df = pd.DataFrame(columns=[e.name for e in estimators] + ['actual_value'])
    for i in range(n_iterations):
        seeds = np.random.randint(np.iinfo(np.int32).max, size=config['n_observation'])
        responses = joblib.Parallel(n_jobs=-1, verbose=50)(
            joblib.delayed(simulate_data)(null_policy, new_policy, environment, item_vectors, seeds[i]) for i in
            range(config['n_observation'])
        )

        sim_data = pd.DataFrame(responses)

        actual_value = sim_data.new_reward.mean()
        estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
        estimated_values['actual_value'] = actual_value
        return_df = return_df.append(estimated_values, ignore_index=True)

    return return_df


if __name__ == "__main__":
    config = {
        "n_items": 40,
        "n_reco": 5,
        "n_observation": 5000,
        "context_dim": 10,
        "tau": 0.01  # almost uniform
    }

    new_item_vectors = np.random.normal(0, 1, size=(config['n_items'], config['context_dim']))
    null_item_vectors = np.random.normal(0, 1, size=(config['n_items'], config['context_dim']))

    # The policy we use to generate sim data
    null_policy = MultinomialPolicy(null_item_vectors, config['n_items'], config['n_reco'], temperature=0.5)

    # The target policy
    new_policy = MultinomialPolicy(new_item_vectors, config['n_items'], config['n_reco'], temperature=0.5)

    # env_item_vectors = np.random.normal(0, 1, size=(config['n_items'], config['context_dim']))
    env_item_vectors = new_item_vectors
    environment = Environment(env_item_vectors, config['context_dim'])
    # environment = AvgEnvironment(env_item_vectors, config['context_dim'])
    # environment = NNEnvironment(env_item_vectors, config['context_dim'])

    reg_pow = np.arange(0, 1)
    reg_params = (10.0 ** reg_pow) / config['n_observation']
    bw_params = [(10.0 ** -1)]
    params = [[r, b1, b2] for r in reg_params for b1 in bw_params for b2 in bw_params]

    seeds = np.random.randint(np.iinfo(np.int32).max, size=config['n_observation'])
    responses = joblib.Parallel(n_jobs=-1, verbose=50)(
        joblib.delayed(simulate_data)(null_policy, new_policy, environment, env_item_vectors, seeds[i]) for i in
        range(config['n_observation'])
    )

    sim_data = pd.DataFrame(responses)

    # ips = IPSEstimator(config['n_reco'], null_policy, new_policy)
    # ips.estimate(sim_data)

    # """
    #  CMEEstimator grid search
    #  """
    # sim_data = simulate_data(null_policy, new_policy, environment, true_context_vector, item_vectors,
    #                          config['n_observation'])
    #
    # cmEstimator = CMEstimator(rbf_kernel, rbf_kernel, None)
    # grid_search_df = grid_search(params, cmEstimator, sim_data, n_iterations=1)
    #
    # grid_search_df.plot.line(x='param', y='estimated_value')
    # print(grid_search_df)

    """
     Comparing between estimators
     """
    estimators = [IPSEstimator(config['n_reco'], null_policy, new_policy),
                  CMEstimator(rbf_kernel, rbf_kernel, params[0]),
                  DirectEstimator()]

    result_df = compare_estimators(estimators, null_policy, new_policy, environment, env_item_vectors, config, 5)
    print(result_df)
    result_df.plot.line(use_index=True)
