from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def simulate(null_policy, new_policy, env, context, context_vectors, item_vectors):
    """
    simulate data given policy, environment and set of context
    :return: an observation
    """
    context = np.random.choice(context)
    # context vector is represented by user feeature
    context_vec = context_vectors[context]

    null_reco = null_policy.recommend(context)
    # recommendation is represented by an average over recommended item vectors
    null_reco_vec = np.mean(np.stack([item_vectors[r] for r in null_reco]), axis=0)
    null_reward, _ = env.get_reward(context, null_reco)

    new_reco = new_policy.recommend(context)
    # recommendation is represented by an average over recommended item vectors
    new_reco_vec = np.mean(np.stack([item_vectors[r] for r in new_reco]), axis=0)
    new_reward, _ = env.get_reward(context, new_reco)

    return {"context": context, "context_vec": context_vec, "null_reco": tuple(null_reco),
            "null_reco_vec": null_reco_vec,
            "null_reward": null_reward, "new_reco": tuple(new_reco), "new_reco_vec": new_reco_vec,
            "new_reward": new_reward}


if __name__ == "__main__":
    config = {
        "n_items": 6,
        "n_reco": 3,
        "n_observation": 5000,
        "context": ['A', 'B', 'C'],
        "n_dim": 5
    }

    result_df = pd.DataFrame(columns=["directVal",
                                      "ipsVal",
                                      "slateVal",
                                      "cmeVal",
                                      "actualVal"
                                      ])

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

    for i in range(5):

        sim_data = list()
        for j in range(config['n_observation']):
            sim_data.append(
                simulate(null_policy, new_policy, environment, config['context'], context_vectors, item_vectors))

        sim_data = pd.DataFrame(sim_data)

        directEstimator = DirectEstimator()
        ipsEstimator = IPSEstimator(config['n_reco'], null_policy, new_policy)
        slateEstimator2 = SlateEstimatorImproved(config['n_reco'], null_policy)

        # Note that we also pass the kernel functions (see scikit-learn doc:
        # sklearn.metrics.pairwise) and their parameters.
        params = [1e-10, 10., 10.]
        cmEstimator = CMEstimator(rbf_kernel, rbf_kernel, params)

        directVal = directEstimator.estimate(sim_data)
        ipsVal = ipsEstimator.estimate(sim_data)
        slateVal = slateEstimator2.estimate(sim_data)
        cmeVal = cmEstimator.estimate(sim_data)

        actualVal = sim_data.new_reward.sum()

        dictVal = {"directVal": directVal,
                   "ipsVal": ipsVal,
                   "slateVal": slateVal,
                   "cmeVal": cmeVal,
                   "actualVal": actualVal}
        result_df = result_df.append(dictVal, ignore_index=True)
        print(result_df)

    result_df.plot.line(use_index=True)
