from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

def simulate_data(null_policy, new_policy, env, context, context_vectors, item_vectors, n_observation):
    """
    simulate data given policy, environment and set of context
    :return: observations
    """

    sim_data = list()
    for _ in range(n_observation):
        pick_context = np.random.choice(context)
        # context vector is represented by user feeature
        context_vec = context_vectors[pick_context]

        null_reco = null_policy.recommend(pick_context)
        # recommendation is represented by an average over recommended item vectors
        null_reco_vec = np.mean(np.stack([item_vectors[r] for r in null_reco]), axis=0)
        null_reward, _ = env.get_reward(pick_context, null_reco)

        new_reco = new_policy.recommend(pick_context)
        # recommendation is represented by an average over recommended item vectors
        new_reco_vec = np.mean(np.stack([item_vectors[r] for r in new_reco]), axis=0)
        new_reward,_ = env.get_reward(pick_context, new_reco)

        observation = {"context": pick_context, "context_vec": context_vec, "null_reco": tuple(null_reco),
                       "null_reco_vec": null_reco_vec,
                       "null_reward": null_reward, "new_reco": tuple(new_reco), "new_reco_vec": new_reco_vec,
                       "new_reward": new_reward}

        sim_data.append(observation)

    sim_data = pd.DataFrame(sim_data)

    return sim_data

def grid_search(params, estimator, sim_data, n_iterations):

    actual_value = sim_data.new_reward.sum()
    return_df = pd.DataFrame(columns=['param', 'estimated_value', 'actual_value', 'error'])
    for param in params:
        estimated_values = []
        for _ in range(n_iterations):
            estimator.params = param
            estimated_values.append(estimator.estimate(sim_data))
        mean_value = np.array(estimated_values).mean()
        ret = {'param': param, 'estimated_value': mean_value, 'actual_value': actual_value,
               'error': np.abs(mean_value - actual_value)}
        return_df = return_df.append(ret, ignore_index=True)

    return return_df


def compare_estimators(estimators, n_iterations, null_policy, new_policy, environment, context_vectors, item_vectors, config):
    
    return_df = pd.DataFrame(columns=[e.name for e in estimators] + ['actual_value'])
    for i in range(n_iterations):
        sim_data = simulate_data(null_policy, new_policy, environment, config['context'], context_vectors, item_vectors,
                                 config['n_observation'])
        actual_value = sim_data.new_reward.sum()
        estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
        estimated_values['actual_value'] = actual_value
        return_df = return_df.append(estimated_values, ignore_index=True)

    return return_df

if __name__ == "__main__":

    config = {
        "n_items": 6,
        "n_reco": 3,
        "n_observation": 5000,
        "context": ['A', 'B', 'C'],
        "n_dim": 5
    }

    # user features and item features, user preference = user_features.dot(item_features)
    context_vectors = dict([(u, np.random.normal(0, 1, config['n_dim'])) for u in config['context']])
    item_vectors = np.stack([np.random.normal(0, 1, config['n_dim']) for u in range(config['n_items'])])

    # soft max distribution of items given the above score
    # this will be used in the Environment, then it is the optimal policy
    context_prob = dict([(u, softmax(np.matmul(s, item_vectors.T))) for (u, s) in context_vectors.items()])

    # null policy is given by soft max distribution of items given negative of above score
    # this is an extreme case where null policy is totally opposite the optimal policy
    null_policy_prob = dict([(u, softmax(-np.matmul(s, item_vectors.T))) for (u, s) in context_vectors.items()])

    null_policy = ProbPolicy(null_policy_prob, config['n_items'], config['n_reco'], greedy=False)
    # The policy we want to estimate is the optimal policy
    new_policy = ProbPolicy(context_prob, config['n_items'], config['n_reco'], greedy=False)

    environment = Environment(context_prob)

    reg_pow = np.arange(-10, 7)
    reg_params = 10.0 ** reg_pow
    params = [[r, 0.1, 0.1] for r in reg_params]

    """
     CMEEstimator grid search
     """
    sim_data = simulate_data(null_policy, new_policy, environment, config['context'], context_vectors, item_vectors,
                             config['n_observation'])
    cmEstimator = CMEstimator(rbf_kernel, rbf_kernel, None)
    grid_search_df = grid_search(params, cmEstimator, sim_data, n_iterations=1)

    grid_search_df.plot.line(x='param', y='estimated_value')
    print(grid_search_df)

    """
     Comparing between estimators
     """
    #estimators = [DirectEstimator(), IPSEstimator(config['n_reco'], null_policy, new_policy),
    #           SlateEstimatorImproved(config['n_reco'], null_policy)]

    #result_df = compare_estimators(estimators, 2, null_policy, new_policy, environment, context_vectors, item_vectors,
    #                                config)
    #
    # result_df.plot.line(use_index=True)
