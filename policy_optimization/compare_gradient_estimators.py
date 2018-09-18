import sys

sys.path.append("../policy_evaluation/")

from CME import *
from Direct import *
from wIPS import *
from DR import *

from ParameterSelector import *
from PolicyGradient import *
from plot_fn import *
import numpy as np

config = {
    "n_users": 100,
    "n_items": 20,
    "context_dim": 10,
    'learning_rate': 0.02
}


def get_expected_var_reward(item_vectors, action_probs, sample_users):
    p = sigmoid(sample_users.dot(item_vectors.T))
    sum_prob = np.sum(p * action_probs, axis=1)
    # using law of total variance
    evpv = np.sum((p * (1 - p)) * action_probs, axis=1)
    vhm = np.sum(action_probs * (p ** 2), axis=1) - sum_prob ** 2

    return sum_prob.mean(), (evpv + vhm).mean()


def run_iteration(user_vectors, item_vectors, null_policy_weight, n_observation, num_iter, est='CME'):
    sample_users = user_vectors[np.random.choice(user_vectors.shape[0], n_observation, True), :]

    null_action_probs = softmax(sample_users.dot(null_policy_weight.T), axis=1)
    null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))
    null_action_vec = item_vectors[null_actions]
    null_rewards = np.random.binomial(1, p=sigmoid(np.sum(sample_users * null_action_vec, axis=1)))

    # decide which estimator to use
    if est == 'Direct':
        null_feature_vec = np.hstack([sample_users, null_action_vec])
        estimator = Direct(null_feature_vec, null_rewards)
    elif est == 'wIPS':
        estimator = wIPS(null_action_probs[np.arange(len(sample_users)), null_actions], null_rewards)
    elif est == 'DR':
        null_feature_vec = np.hstack([sample_users, null_action_vec])
        estimator = DR(null_feature_vec, null_action_probs[np.arange(len(sample_users)), null_actions], null_rewards)
    elif est == 'CME':
        estimator = CME(sample_users, null_action_vec, null_rewards)
    else:
        sys.exit(1)

    target_cme_rewards = []
    target_exp_rewards = []
    target_var_rewards = []

    policy_grad_graph = tf.Graph()
    sess = tf.Session(graph=policy_grad_graph)
    with policy_grad_graph.as_default():
        policy_grad = PolicyGradientAgent(config, sess, null_policy_weight)
        sess.run(tf.global_variables_initializer())

    for i in range(num_iter):

        target_actions, target_action_probs = policy_grad.act(sample_users)
        target_actions = target_actions.nonzero()[2]
        target_action_vec = item_vectors[target_actions]

        train_actions = null_actions
        # estimation
        if est == 'Direct':
            target_feature_vec = np.hstack([sample_users, target_action_vec])
            target_reward_vec = estimator.estimate(target_feature_vec)
            train_actions = target_actions
        elif est == 'wIPS':
            target_reward_vec = estimator.estimate(
                target_action_probs[np.arange(len(sample_users)), target_actions])
        elif est == 'DR':
            target_feature_vec = np.hstack([sample_users, target_action_vec])
            target_reward_vec = estimator.estimate(target_feature_vec, target_action_probs[
                np.arange(len(sample_users)), target_actions])
            train_actions = target_actions
        elif est == 'CME':
            target_reward_vec = estimator.estimate(sample_users, target_action_vec)

        target_reward = target_reward_vec.mean()
        expected_reward, var_reward = get_expected_var_reward(item_vectors, target_action_probs, sample_users)
        target_cme_rewards.append(target_reward)
        target_exp_rewards.append(expected_reward)
        target_var_rewards.append(var_reward / n_observation)

        loss = policy_grad.train_step(sample_users, train_actions, target_reward_vec)

        if i % 20 == 0:
            print("iter {}, Expected reward: {}".format(i, expected_reward))
            print("iter {}, CME reward: {}".format(i, target_reward))
            print("iter {}, loss: {}".format(i, loss))

    sess.close()

    optimal_actions = np.argmax(sample_users.dot(item_vectors.T), axis=1)
    optimal_action_probs = np.zeros((optimal_actions.shape[0], config['n_items']))
    optimal_action_probs[np.arange(optimal_actions.shape[0]), optimal_actions] = 1
    optimal_reward, _ = get_expected_var_reward(item_vectors, optimal_action_probs, sample_users)

    return target_exp_rewards, target_var_rewards, target_cme_rewards, optimal_reward


### SIMULATION STARTS HERE ###

np.random.seed(321)

user_vectors = np.random.normal(0, 1.0, size=(config['n_users'], config['context_dim']))
item_vectors = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim']))

null_policy_weight = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim']))

num_iter = 500
estimators = ['CME', 'Direct', 'wIPS']
exp_rewards = np.zeros((len(estimators), num_iter))
var_rewards = np.zeros((len(estimators), num_iter))

for n_obs in [5000]:

    for i in range(len(estimators)):
        exp_rewards[i], var_rewards[i], cme_reward, optimal_reward = \
            run_iteration(user_vectors,
                          item_vectors,
                          null_policy_weight,
                          n_obs,
                          num_iter,
                          estimators[i])

    plot_comparison_result(exp_rewards,
                           var_rewards,
                           optimal_reward,
                           "policy_optimization/_result/compare_est_n_obs_{}.pdf".format(n_obs),
                           "Comparison",
                           estimators)
