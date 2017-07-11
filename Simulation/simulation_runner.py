from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

def simulate(policy, env, users):
    """
    simulate data given policy, environment and set of users
    :return: list of observations {"x": context, "y": recommendation, "r": reward, "p": pick item}
    """
    context = np.random.choice(users)
    reco = policy.recommend(context)
    (reward, pick) = env.get_reward(context, reco)
    return {"x": context, "y": tuple(reco), "r": reward, "p": pick}

def estimate(estimator, policy, sim_data):
    """
    estimate reward for a given policy and estimator
    :param sim_data: list of observations {"x": context, "y": recommendation, "r": reward, "p": pick item}
    :return: sum(reward)
    """
    return sum([estimator.estimate(d['x'], np.array(d['y']), d['r'], policy.recommend(d['x'])) for d in sim_data])

def get_sim_reward(policy, env, users, N=10000):
    """
    get actual reward from simulation for a given policy and environment
    :param N: number of simulated observations
    :return: sum(actual_reward)
    """
    sim = list()
    for i in range(N):
        sim.append(simmulate(policy, env, users))
    return sum([d['r'] for d in sim])

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
                                      "cmeVal",
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

    for i in range(20):

        null_policy = ProbPolicy(null_policy_prob, config['n_items'], config['n_reco'], greedy=False)
        # The policy we want to estimate is the optimal policy
        new_policy = ProbPolicy(user_prob, config['n_items'], config['n_reco'], greedy=False)

        environment = Environment(user_prob)

        sim_data = list()
        for i in range(config['n_observation']):
            sim_data.append(simmulate(null_policy, environment, config['users']))

        directEstimator = DirectEstimator(config['n_reco'], sim_data)
        ipsEstimator = IPSEstimator(config['n_reco'], null_policy, new_policy)
        slateEstimator2 = SlateEstimatorImproved(config['n_reco'], null_policy)

        # we instantiate the counterfactual mean estimator here. Note that we also pass the
        # kernel functions (see scikit-learn doc: sklearn.metrics.pairwise) and their parameters.
        
        #cmEstimator = CMEstimator(config['n_reco'], null_policy, sim_data, rbf_kernel, rbf_kernel, ...)
        
        directVal = estimate(directEstimator, new_policy, sim_data)
        ipsVal = estimate(ipsEstimator, new_policy, sim_data)
        slateVal = estimate(slateEstimator2, new_policy, sim_data)
        #cmeVal = estimate(cmEstimator, new_policy, sim_data)
        
        actualVal = get_sim_reward(new_policy, environment, config['users'], config['n_observation'])

        dictVal = {"directVal": directVal,
                   "ipsVal": ipsVal,
                   "slateVal": slateVal,
                   "cmeVal": cmeVal,
                   "actualVal": actualVal}
        result_df = result_df.append(dictVal, ignore_index=True)
        print(result_df)

    result_df.plot.line(use_index=True)
