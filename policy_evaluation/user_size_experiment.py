from Environment import *
from Estimator import *
from Policy import *
from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
# import matplotlib.pyplot as plt
import joblib
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def simulate_data(null_policy, target_policy, environment, item_vectors):
    """
    simulate data given policy, environment and set of context
    :return: observations
    """
    user = environment.get_context()
    null_reco, null_multinomial, null_user_vector = null_policy.recommend(user)
    # recommendation is represented by a concatenation of recommended item vectors
    # null_reco_vec = np.mean(item_vectors[null_reco], axis=0)
    null_reco_vec = np.concatenate(item_vectors[null_reco])
    null_reward = environment.get_reward(user, null_reco)

    target_reco, target_multinomial, _ = target_policy.recommend(user)
    # recommendation is represented by a concatenation of recommended item vectors
    # target_reco_vec = np.mean(item_vectors[target_reco], axis=0)
    target_reco_vec = np.concatenate(item_vectors[target_reco])
    target_reward = environment.get_reward(user, target_reco)

    observation = {"null_context_vec": null_user_vector, "target_context_vec": null_user_vector,
                   "null_reco": tuple(null_reco),
                   "null_reco_vec": null_reco_vec, "null_reward": null_reward,
                   "target_reco": tuple(target_reco), "null_multinomial": null_multinomial,
                   "target_multinomial": target_multinomial, "target_reco_vec": target_reco_vec,
                   "target_reward": target_reward, "user": user}

    return observation


def get_actual_reward(target_policy, environment, n=100000):
    sum_reward = 0
    for i in range(n):
        user = environment.get_context()
        target_reco, target_multinomial, _ = target_policy.recommend(user)
        sum_reward += environment.get_reward(user, target_reco)

    return sum_reward / float(n)


def grid_search(params, estimator, sim_data, n_iterations):
    """
    :param params:
    :param estimator:
    :param sim_data:
    :param n_iterations:
    :return:
    """
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


def compare_estimators(estimators, null_policy, target_policy, environment, item_vectors, config, seed):
    np.random.seed(seed)
    sim_data = [simulate_data(null_policy, target_policy, environment, item_vectors)
                for _ in range(config['n_observation'])]
    sim_data = pd.DataFrame(sim_data)

    # parameter selection
    direct_selector = ParameterSelector(estimators[2])  # direct estimator
    params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
    direct_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[2] = direct_selector.estimator

    estimators[3].params = direct_selector.parameters  # doubly robust estimator

    cme_selector = ParameterSelector(estimators[4])  # cme estimator
    params_grid = [[(10.0 ** p) / config['n_observation'], 1.0, 1.0] for p in np.arange(-6, 0, 1)]
    cme_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[4] = cme_selector.estimator

    actual_value = get_actual_reward(target_policy, environment)

    estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
    estimated_values['actual_value'] = actual_value
    estimated_values['null_reward'] = sim_data.null_reward.mean()

    for e in estimators:
        estimated_values[e.name + '_square_error'] = \
            (estimated_values[e.name] - estimated_values['actual_value']) ** 2
    print(estimated_values)

    return estimated_values


def compare_kernel_regression(estimators, null_policy, target_policy, environment, item_vectors, config, seed):
    np.random.seed(seed)
    sim_data = [simulate_data(null_policy, target_policy, environment, item_vectors)
                for _ in range(config['n_observation'])]
    sim_data = pd.DataFrame(sim_data)

    # parameter selection
    direct_selector = ParameterSelector(estimators[0])  # direct estimator
    params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
    direct_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[0] = direct_selector.estimator

    direct_selector = ParameterSelector(estimators[1])  # direct estimator
    params_grid = [0.001, .01, .1, 1, 10]
    direct_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[1] = direct_selector.estimator

    cme_selector = ParameterSelector(estimators[2])  # cme estimator
    params_grid = [[(10.0 ** p) / config['n_observation'], 1.0, 1.0] for p in np.arange(-6, 0, 1)]
    cme_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[2] = cme_selector.estimator

    actual_value = get_actual_reward(target_policy, environment)

    estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
    estimated_values['actual_value'] = actual_value
    estimated_values['null_reward'] = sim_data.null_reward.mean()

    for e in estimators:
        estimated_values[e.name + '_square_error'] = \
            (estimated_values[e.name] - estimated_values['actual_value']) ** 2
    print(estimated_values)

    return estimated_values


if __name__ == "__main__":

    try:
        # get an index of a multiplier as an argument
        num_users = 10 + 20 * int(sys.argv[1])
    except:
        sys.exit(1)
        # num_users = 100

    config = {
        "n_users": num_users,
        "n_items": 20,
        "n_reco": 5,
        "n_observation": 5000,
        "context_dim": 10
    }

    result_df = pd.DataFrame()
    multiplier = -0.3
    num_iter = 5

    user_vectors = np.random.normal(0, 1, size=(config['n_users'], config['context_dim']))
    target_user_vectors = user_vectors * np.random.binomial(1, 0.5, size=user_vectors.shape)
    item_vectors = np.random.normal(0, 1, size=(config['n_items'], config['context_dim']))

    null_user_vectors = multiplier * target_user_vectors

    # The policy we use to generate sim data
    null_policy = MultinomialPolicy(item_vectors, null_user_vectors, config['n_items'], config['n_reco'],
                                    temperature=0.5, cal_gamma=False)

    # The target policy
    target_policy = MultinomialPolicy(item_vectors, target_user_vectors, config['n_items'], config['n_reco'],
                                      temperature=1.0, cal_gamma=False)

    environment = AvgEnvironment(item_vectors, user_vectors)

    reg_pow = -1
    reg_params = (10.0 ** reg_pow) / config['n_observation']
    bw_params = (10.0 ** 0)
    params = [reg_params, bw_params, bw_params]

    """ 
     Comparing between estimators
     """
    # estimators = [IPSEstimator(config['n_reco'], null_policy, target_policy),
    #               SlateEstimator(config['n_reco'], null_policy),
    #               DirectEstimator(),
    #               DoublyRobustEstimator(config['n_reco'], null_policy, target_policy),
    #               CMEstimator(rbf_kernel, rbf_kernel, params)]
    estimators = [DirectEstimator(),
                  DirectKernelEstimator(),
                  CMEstimator(rbf_kernel, rbf_kernel, params)]

    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_iter)
    compare_df = joblib.Parallel(n_jobs=1, verbose=50)(
        joblib.delayed(compare_kernel_regression)(estimators, null_policy, target_policy, environment, item_vectors,
                                                  config, seeds[i]) for i in range(num_iter)
    )
    compare_df = pd.DataFrame(compare_df)
    compare_df['num_users'] = num_users
    result_df = result_df.append(compare_df, ignore_index=True)

    # compare_df[list(filter(lambda x: 'error' not in x,compare_df.columns))].plot()
    result_df.to_csv("usersize_kernel_result_%d.csv" % (num_users), index=False)
