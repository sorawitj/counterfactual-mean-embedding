import sys

from CME import *
from Direct import *
from wIPS import *
from DR import *

from ParameterSelector import *
from PolicyGradient import *
from plot_fn import *
import numpy as np

import pickle

config = {
    "n_users": 100,
    "n_items": 20,
    "context_dim": 10,
    'learning_rate': 0.025
}


def get_expected_var_reward(item_vectors, action_probs, sample_users):
    p = sigmoid(sample_users.dot(item_vectors.T))
    sum_prob = np.sum(p * action_probs, axis=1)
    # using law of total variance
    evpv = np.sum((p * (1 - p)) * action_probs, axis=1)
    vhm = np.sum(action_probs * (p ** 2), axis=1) - sum_prob ** 2

    return sum_prob.mean(), (evpv + vhm).mean()


def run_iteration(sample_users, item_vectors, null_policy_weight, n_observation, num_iter, est='CME'):
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

    target_pred_rewards = []
    target_exp_rewards = []
    target_var_rewards = []

    policy_grad_graph = tf.Graph()
    sess = tf.Session(graph=policy_grad_graph)
    with policy_grad_graph.as_default():
        policy_grad = PolicyGradientAgent(config, sess, null_policy_weight.T)
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
        target_pred_rewards.append(target_reward)
        target_exp_rewards.append(expected_reward)
        target_var_rewards.append(var_reward / n_observation)

        loss = policy_grad.train_step(sample_users, train_actions, target_reward_vec)

        if i % 20 == 0:
            print("iter {}, Expected reward: {}".format(i, expected_reward))
            print("iter {}, Predicted reward: {}".format(i, target_reward))
            print("iter {}, loss: {}".format(i, loss))

    sess.close()

    optimal_actions = np.argmax(sample_users.dot(item_vectors.T), axis=1)
    optimal_action_probs = np.zeros((optimal_actions.shape[0], config['n_items']))
    optimal_action_probs[np.arange(optimal_actions.shape[0]), optimal_actions] = 1
    optimal_reward, _ = get_expected_var_reward(item_vectors, optimal_action_probs, sample_users)

    return target_exp_rewards, target_var_rewards, target_pred_rewards, optimal_reward


### SIMULATION STARTS HERE ###

if __name__ == "__main__":

    try:
        # get an index of a multiplier as an argument
        estimator_index = int(sys.argv[1])
    except:
        sys.exit(1)

    np.random.seed(321)

    user_components = np.random.choice(5, size=config['n_users'], p=(0.3, 0.1, 0.3, 0.1, 0.2), replace=True)
    item_components = np.random.choice(3, size=config['n_items'], p=(0.3, 0.5, 0.2), replace=True)

    mu_users = np.array([1, -1, 3, -2, 0])
    sd_users = np.array([1, -1, 3, -2, 0])
    mu_items = np.array([0.1, 1, 3, 2, 1])
    sd_items = np.array([1, 0.1, 2])

    user_vectors = np.random.normal(0, 1.0, size=(config['n_users'], config['context_dim'])) \
               * np.expand_dims(sd_users[user_components], 1) + np.expand_dims(mu_users[user_components], 1)
    item_vectors = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim'])) \
               * np.expand_dims(sd_items[item_components], 1) + np.expand_dims(mu_items[item_components], 1)

    # create random null policy
    # null_policy_weight = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim']))

    # create null policy which is different from the optimal policy
    null_policy_weight = -.3 * item_vectors

    num_iter = 300
    num_obs = 500
    estimators = ['Direct', 'CME', 'wIPS']
    exp_rewards = np.zeros((1, num_iter))
    pred_rewards = np.zeros((1, num_iter))
    var_rewards = np.zeros((1, num_iter))

    sample_users = user_vectors[np.random.choice(user_vectors.shape[0], num_obs, True), :]

    exp_rewards, var_rewards, pred_rewards, optimal_reward = \
      run_iteration(sample_users,
                        item_vectors,
                        null_policy_weight,
                        num_obs,
                        num_iter,
                        estimators[estimator_index])

    # save the results
    with open("../compare_estimators_result/compare_{}_random_n_obs_{}.pickle".format(estimators[estimator_index],num_obs), 'wb') as handle:
        pickle.dump(exp_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pred_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(var_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(optimal_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
