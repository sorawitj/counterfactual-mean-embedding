from CME import *
from ParameterSelector import *
from PolicyGradient import *
import numpy as np
# import matplotlib.pyplot as plt
import joblib
import os

config = {
    "n_users": 10,
    "n_items": 5,
    "n_reco": 4,
    "n_observation": 3000,
    "context_dim": 10,
    'learning_rate': 0.1
}

user_item_vectors = np.random.normal(0, 1, size=(config['n_users'], config['n_items'], config['context_dim']))

true_weights = np.random.normal(0, 1, config['context_dim'])
null_policy_weight = np.random.normal(0, 1, config['context_dim'])

sample_users = user_item_vectors[np.random.choice(user_item_vectors.shape[0], config['n_observation'], True), :]

optimal_action_probs = softmax(sample_users.dot(true_weights.T), axis=1)
null_action_probs = softmax(sample_users.dot(null_policy_weight.T), axis=1)

null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))
optimal_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), optimal_action_probs)))

null_rewards = np.random.binomial(1, p=sigmoid(
    sample_users[np.arange(len(sample_users)), null_actions, :].dot(true_weights.T)))
# null_rewards = sample_users[np.arange(len(sample_users)), null_actions, :].dot(true_weights.T)
optimal_rewards = np.random.binomial(1, p=sigmoid(
    sample_users[np.arange(len(sample_users)), optimal_actions, :].dot(true_weights.T)))
# optimal_rewards = sample_users[np.arange(len(sample_users)), optimal_actions, :].dot(true_weights.T)

sess = tf.Session()
policy_grad = PolicyGradientAgent(config, sess, null_policy_weight)
sess.run(tf.global_variables_initializer())

null_feature_vec = sample_users[np.arange(len(sample_users)), null_actions, :]
cmeEstimator = CME(null_feature_vec, null_rewards)

for _ in range(200):
    target_actions = policy_grad.act(sample_users)

    target_feature_vec = sample_users[np.arange(len(sample_users)), target_actions, :]

    target_reward = cmeEstimator.estimate(target_feature_vec)
    print(target_reward.sum())

    # target_reward = np.random.binomial(1, p=sigmoid(sample_users[np.arange(len(sample_users)), target_actions, :].dot(true_weights.T)))
    # print(target_reward.mean())

    policy_grad.train_step(sample_users, null_actions, target_reward)

target_rewards = np.random.binomial(1, p=sigmoid(
    sample_users[np.arange(len(sample_users)), target_actions, :].dot(true_weights.T)))
print("rewards")
print(target_rewards.mean())
print(optimal_rewards.mean())
