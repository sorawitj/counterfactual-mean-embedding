from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def simulate(null_policy, new_policy, env, users):
    """
    simulate data given policy, environment and set of users
    :return: an observation
    """
    context = np.random.choice(users)
    null_reco = null_policy.recommend(context)
    null_reward, _ = env.get_reward(context, null_reco)

    new_reco = new_policy.recommend(context)
    new_reward, _ = env.get_reward(context, new_reco)

    return {"context": context, "null_reco": tuple(null_reco), "null_reward": null_reward, "new_reco": tuple(new_reco),
            "new_reward": new_reward}


if __name__ == "__main__":
    config = {
        "n_items": 6,
        "n_reco": 3,
        "n_observation": 5000,
        "users": ['A', 'B', 'C']
    }

    result_df = pd.DataFrame(columns=["directVal",
                                      "ipsVal",
                                      "slateVal",
                                      # "cmeVal",
                                      "actualVal"
                                      ])

    # score of user preferences (high score means user likes that item)
    user_score = [(u, np.random.normal(0, 1, config['n_items'])) for u in config['users']]

    # soft max distribution of items given the above score
    # this will be used in the Environment, then it is the optimal policy
    user_prob = dict([(u, softmax(s)) for (u, s) in user_score])

    # null policy is given by soft max distribution of items given negative of above score
    # this is an extreme case where null policy is totally opposite the optimal policy
    null_policy_prob = dict([(u, softmax(-s)) for (u, s) in user_score])

    null_policy = ProbPolicy(null_policy_prob, config['n_items'], config['n_reco'], greedy=False)
    # The policy we want to estimate is the optimal policy
    new_policy = ProbPolicy(user_prob, config['n_items'], config['n_reco'], greedy=False)

    environment = Environment(user_prob)

    for i in range(5):

        sim_data = list()
        for i in range(config['n_observation']):
            sim_data.append(simulate(null_policy, new_policy, environment, config['users']))

        sim_data = pd.DataFrame(sim_data)

        directEstimator = DirectEstimator()
        ipsEstimator = IPSEstimator(config['n_reco'], null_policy, new_policy)
        slateEstimator2 = SlateEstimatorImproved(config['n_reco'], null_policy)

        # we instantiate the counterfactual mean estimator here. Note that we also pass the
        # kernel functions (see scikit-learn doc: sklearn.metrics.pairwise) and their parameters.

        # cmEstimator = CMEstimator(config['n_reco'], null_policy, sim_data, rbf_kernel, rbf_kernel, ...)

        directVal = directEstimator.estimate(sim_data)
        ipsVal = ipsEstimator.estimate(sim_data)
        slateVal = slateEstimator2.estimate(sim_data)
        # cmeVal = estimate(cmEstimator, new_policy, sim_data)

        actualVal = sim_data.new_reward.sum()

        dictVal = {"directVal": directVal,
                   "ipsVal": ipsVal,
                   "slateVal": slateVal,
                   # "cmeVal": cmeVal,
                   "actualVal": actualVal}
        result_df = result_df.append(dictVal, ignore_index=True)
        print(result_df)

    result_df.plot.line(use_index=True)
