get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import sys

sys.path.append("Simulation")
from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd

"""On develop, please ignore this"""

def simmulate(policy, env, users):
    x = np.random.choice(users)
    reco = policy.recommend(x)
    (reward, pick) = env.get_reward(x, reco)
    return {"x": x, "y": tuple(reco), "r": reward, "h": pick}


def estimate(estimator, policy, sim_data):
    return sum(
        [estimator.estimate(d['x'], np.array(d['y']), d['r'], policy.recommend(d['x'])) for d in
         sim_data])


def getSimVal(policy, env, users, N=10000):
    sim = list()
    for i in range(N):
        sim.append(simmulate(policy, env, users))
    return sum([d['r'] for d in sim])


def getBHR(policy, sim_data):
    return sum([d['h'] in policy.recommend(d['x']) for d in sim_data if d['h'] is not None])


if __name__ == "__main__":
    n_reco = 3
    n_hotel_per_group = 5
    n = 5000
    n_hotel = n_hotel_per_group * 2
    users = ["A"]
    d_policy = {"A": np.array([0, 7, 1])}
    m_policy = {"A": np.array([0, 1, 2])}

    res = []
    for i in range(100):
        null_policy_prob = np.repeat([2.0, 1.0], n_hotel_per_group)
        null_policy_prob /= null_policy_prob.sum()

        nullPolicy = ProbPolicy({"A": null_policy_prob}, n_reco=n_reco,
                                n_items=n_hotel, greedy=False)

        env = BinaryDiversEnvironment(0.5, 0.25, (0.6, 0.4), n_hotel_per_group)

        sim_data = [simmulate(nullPolicy, env, ["A"]) for _ in range(n)]

        slateEstimator = SlateEstimator(n_reco, nullPolicy)
        slateEstimator2 = SlateEstimatorImproved(n_reco, nullPolicy)

        mostCommon = FixedPolicy(m_policy)
        bhrMostCommon = getBHR(mostCommon, sim_data)
        slateMostCommon = estimate(slateEstimator, mostCommon, sim_data)
        slateMostCommon2 = estimate(slateEstimator2, mostCommon, sim_data)
        actMostCommon = getSimVal(mostCommon, env, ["A"], n)

        dPolicy = FixedPolicy(d_policy)
        bhrUser = getBHR(dPolicy, sim_data)
        slateUser = estimate(slateEstimator, dPolicy, sim_data)
        slateUser2 = estimate(slateEstimator2, dPolicy, sim_data)
        actUser = getSimVal(dPolicy, env, users, n)

        print(i)

        dictVal = {"most_common_bhr": bhrMostCommon,
                   "diver_bhr": bhrUser,
                   "most_common_slate": slateMostCommon,
                   "most_common_slate2": slateMostCommon2,
                   "diver_slate": slateUser,
                   "diver_slate2": slateUser2,
                   "most_common_act": actMostCommon,
                   "diver_act": actUser}

        print(pd.Series(dictVal).sort_index())
        res.append(dictVal)

resDf = pd.DataFrame(res)

resDf.groupby(np.ones(len(resDf))).agg(
    lambda x: dict(
        mean=np.mean(x),
        stdErr=np.std(x) / np.sqrt(x.count())
    )
)
res1 = res
