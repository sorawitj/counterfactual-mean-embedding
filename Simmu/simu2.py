get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import sys

sys.path.append("Simmu")
from Environment import *
from Estimator import *
from Policy import *
from Utils import *
import numpy as np
import pandas as pd


def simmulate(policy, env, users):
    x = np.random.choice(users)
    reco = policy.recommend(x)
    (reward, pick) = env.get_reward(x, reco)
    return {"x": x, "y": tuple(reco), "r": reward, "h": pick}


def estimate(estimator, policy, simData):
    return sum(
        [estimator.estimate(d['x'], np.array(d['y']), d['r'], policy.recommend(d['x'])) for d in
            simData])


def getSimVal(policy, env, users, N=10000):
    sim = list()
    for i in xrange(N):
        sim.append(simmulate(policy, env, users))
    return sum([d['r'] for d in sim])


def getBHR(policy, simData):
    return sum([d['h'] in policy.recommend(d['x']) for d in simData if d['h'] is not None])


if __name__ == "__main__":
    n_reco = 3
    n_hotel_per_group = 5
    n = 5000
    n_hotel = n_hotel_per_group * 2
    users = ["A"]
    d_policy = {"A": np.array([0,7,1])}
    m_policy = {"A": np.array([0,1,2])}

    res = []
    for i in xrange(100):
        null_policy_prob = np.repeat([2.0, 1.0], n_hotel_per_group)
        null_policy_prob /= null_policy_prob.sum()


        nullPolicy = PriorPolicy({"A": null_policy_prob}, n_reco=n_reco,
            n_hotels=n_hotel, greedy=False)

        env = BinaryDiversEnvironment(0.5, 0.25, (0.6, 0.4), n_hotel_per_group)

        simData = [simmulate(nullPolicy, env, ["A"]) for _ in xrange(n)]

        slateEstimator = SlateEstimator(n_reco, nullPolicy)
        slateEstimator2 = SlateEstimator2(n_reco, nullPolicy)

        mostCommon = FixedPolicy(m_policy)
        bhrMostCommon = getBHR(mostCommon, simData)
        slateMostCommon = estimate(slateEstimator, mostCommon, simData)
        slateMostCommon2 = estimate(slateEstimator2, mostCommon, simData)
        actMostCommon = getSimVal(mostCommon, env, ["A"], n)

        dPolicy = FixedPolicy(d_policy)
        bhrUser = getBHR(dPolicy, simData)
        slateUser = estimate(slateEstimator, dPolicy, simData)
        slateUser2 = estimate(slateEstimator2, dPolicy, simData)
        actUser = getSimVal(dPolicy, env, users, n)

        print i

        dictVal = {"most_common_bhr": bhrMostCommon,
            "diver_bhr": bhrUser,
            "most_common_slate": slateMostCommon,
            "most_common_slate2": slateMostCommon2,
            "diver_slate": slateUser,
            "diver_slate2": slateUser2,
            "most_common_act": actMostCommon,
            "diver_act": actUser}

        print pd.Series(dictVal).sort_index()
        res.append(dictVal)



resDf = pd.DataFrame(res)

resDf.groupby(np.ones(len(resDf))).agg(
    lambda x: dict(
        mean = np.mean(x),
        stdErr = np.std(x)/np.sqrt(x.count())
    )
)
res1 = res