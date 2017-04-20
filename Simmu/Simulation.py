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
    return sum([estimator.estimate(d['x'], np.array(d['y']), d['r'], policy.recommend(d['x'])) for d in simData])


def getSimVal(policy, env, users, N=10000):
    sim = list()
    for i in range(N):
        sim.append(simmulate(policy, env, users))
    return sum([d['r'] for d in sim])


def getBHR(policy, simData):
    return sum([d['h'] in policy.recommend(d['x']) for d in simData if d['h'] is not None])


if __name__ == "__main__":
    n_hotel = 6
    n_reco = 3
    n = 5000
    users = ["A", "B", "C"]

    resDf = pd.DataFrame(columns=["directVal",
                                  "ipsVal",
                                  "slateVal",
                                  "actualVal"
                                  ])

    null_score = [(u, np.random.normal(0, 1, n_hotel)) for u in users]
    user_score = [(u, np.random.normal(0, 1, n_hotel)) for u in users]
    # user_score = [(u, np.repeat([1.0], [n_hotel])) for u in users]
    # policy_prob = dict([(u, softmax(-s * 1.25)) for (u, s) in user_score])
    policy_prob = dict([(u, softmax(s * 1.25)) for (u, s) in null_score])
    user_prob = dict([(u, softmax(s)) for (u, s) in user_score])

    for i in range(50):

        nullPolicy = PriorPolicy(policy_prob, n_reco=n_reco, n_hotels=n_hotel, greedy=False)
        newPolicy = PriorPolicy(user_prob, n_hotel, n_reco, greedy=False)

        env = Environment(user_prob)

        simData = list()
        for i in range(n):
            simData.append(simmulate(nullPolicy, env, users))

        directEstimator = DirectEstimator(n_reco, newPolicy, simData)
        ipsEstimator = IPSEstimator(n_reco, nullPolicy, newPolicy)
        slateEstimator2 = SlateEstimator2(n_reco, nullPolicy)

        directVal = estimate(directEstimator, newPolicy, simData)
        ipsVal = estimate(ipsEstimator, newPolicy, simData)
        slateVal = estimate(slateEstimator2, newPolicy, simData)

        actualVal = getSimVal(newPolicy, env, users, n)

        print("directVal: %s" % directVal)
        print("ipsVal: %s" % ipsVal)
        print("slateVal: %s" % slateVal)
        print("actualVal: %s" % actualVal)

        dictVal = {"directVal": directVal,
                   "ipsVal": ipsVal,
                   "slateVal": slateVal,
                   "actualVal": actualVal}
        resDf = resDf.append(dictVal, ignore_index=True)
        print(resDf)
