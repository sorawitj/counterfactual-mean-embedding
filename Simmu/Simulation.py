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

    resDf = pd.DataFrame(columns=["bhr_most_common",
                                  "bhr_user_pref",
                                  "slate_most_common",
                                  "slate_most_common2",
                                  "slate_user_pref",
                                  "slate_user_pref2",
                                  "act_most_common",
                                  "act_user_pref"
                                  ])

    # user_score = [(u, np.random.normal(0, 1, n_hotel)) for u in users]
    user_score = [(u, np.repeat([1.0], [n_hotel])) for u in users]
    # policy_prob = dict([(u, softmax(-s * 1.25)) for (u, s) in user_score])
    policy_prob = dict([(u, softmax(-s * 1.25)) for (u, s) in user_score])
    user_prob = dict([(u, softmax(s)) for (u, s) in user_score])

    for i in range(50):

        nullPolicy = PriorPolicy(policy_prob, n_reco=n_reco, n_hotels=n_hotel, greedy=False)
        env = Environment(user_prob)

        simData = list()
        for i in range(n):
            simData.append(simmulate(nullPolicy, env, users))

        slateEstimator = SlateEstimator(n_reco, nullPolicy)
        slateEstimator2 = SlateEstimator2(n_reco, nullPolicy)

        mostCommon = MostCommonByUserPolicy(n_hotel, n_reco)
        mostCommon.train(simData)
        bhrMostCommon = getBHR(mostCommon, simData)
        slateMostCommon = estimate(slateEstimator, mostCommon, simData)
        slateMostCommon2 = estimate(slateEstimator2, mostCommon, simData)
        actMostCommon = getSimVal(mostCommon, env, users, n)

        userPolicy = PriorPolicy(user_prob, n_hotel, n_reco, greedy=True)
        bhrUser = getBHR(userPolicy, simData)
        slateUser = estimate(slateEstimator, userPolicy, simData)
        slateUser2 = estimate(slateEstimator2, userPolicy, simData)
        actUser = getSimVal(userPolicy, env, users, n)

        print("bhrUser: %s" % bhrUser)
        print("slateUser: %s" % slateUser)
        print("actUser: %s" % actUser)

        dictVal = {"bhr_most_common": bhrMostCommon,
                   "bhr_user_pref": bhrUser,
                   "slate_most_common": slateMostCommon,
                   "slate_most_common2": slateMostCommon2,
                   "slate_user_pref": slateUser,
                   "slate_user_pref2": slateUser2,
                   "act_most_common": actMostCommon,
                   "act_user_pref": actUser}
        resDf = resDf.append(dictVal, ignore_index=True)
        print(resDf)
