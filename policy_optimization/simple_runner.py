from CME import *
from ParameterSelector import *
from PolicyGradient import *
from plot_fn import *
import numpy as np

config = {
    "n_users": 100,
    "n_items": 5,
    "context_dim": 10,
    'learning_rate': 0.05
}


def get_sample_rewards(weight_vector, actions, sample_users):
    return np.random.binomial(1, p=sigmoid(
        sample_users[np.arange(len(sample_users)), actions, :].dot(weight_vector.T)))


def get_expected_var_reward(weight_vector, action_probs, sample_users):
    p = sigmoid(sample_users.dot(weight_vector))
    sum_prob = np.sum(p * action_probs, axis=1)
    # using law of total variance
    evpv = np.sum((p * (1 - p)) * action_probs, axis=1)
    vhm = np.sum(action_probs * (p ** 2), axis=1) - sum_prob ** 2

    return sum_prob.mean(), (evpv + vhm).mean()


def run_iteration(user_item_vectors, true_weights, null_policy_weight, n_observation):
    sample_users = user_item_vectors[np.random.choice(user_item_vectors.shape[0], n_observation, True), :]

    null_action_probs = softmax(sample_users.dot(null_policy_weight.T), axis=1)

    null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))

    null_rewards = get_sample_rewards(true_weights, null_actions, sample_users)

    with tf.Graph().as_default(), tf.Session() as sess:
        policy_grad = PolicyGradientAgent(config, sess, null_policy_weight)
        sess.run(tf.global_variables_initializer())

        null_feature_vec = sample_users[np.arange(len(sample_users)), null_actions, :]
        cme_estimator = CME(null_feature_vec, null_rewards)

        target_cme_rewards = []
        target_exp_rewards = []
        target_var_rewards = []

        for i in range(200):
            target_actions, target_action_probs = policy_grad.act(sample_users)
            target_actions = target_actions.nonzero()[2]

            target_feature_vec = sample_users[np.arange(len(sample_users)), target_actions, :]

            target_reward_vec = cme_estimator.estimate(target_feature_vec)
            target_reward = target_reward_vec.sum()
            expected_reward, var_reward = get_expected_var_reward(true_weights, target_action_probs, sample_users)
            target_cme_rewards.append(target_reward)
            target_exp_rewards.append(expected_reward)
            target_var_rewards.append(var_reward / n_observation)

            loss = policy_grad.train_step(sample_users, null_actions, target_reward_vec)

            if i % 20 == 0:
                print("iter {}, Expected reward: {}".format(i, expected_reward))
                print("iter {}, CME reward: {}".format(i, target_reward))
                print("iter {}, loss: {}".format(i, loss))

    optimal_actions = np.argmax(sample_users.dot(true_weights.T), axis=1)
    optimal_action_probs = np.zeros((optimal_actions.shape[0], sample_users.shape[1]))
    optimal_action_probs[np.arange(optimal_actions.shape[0]), optimal_actions] = 1
    optimal_reward, _ = get_expected_var_reward(true_weights, optimal_action_probs, sample_users)

    return target_exp_rewards, target_var_rewards, target_cme_rewards, optimal_reward


np.random.seed(321)

user_item_vectors = np.random.normal(0, 1, size=(config['n_users'], config['n_items'], config['context_dim']))
true_weights = np.random.normal(0, 1, config['context_dim'])
null_policy_weight = np.random.normal(0, 1, config['context_dim'])

for n_obs in [1000, 5000, 10000]:
    exp_reward, var_reward, cme_reward, optimal_reward = \
        run_iteration(user_item_vectors,
                      true_weights,
                      null_policy_weight,
                      n_obs)

    plot_result(exp_reward,
                cme_reward,
                var_reward,
                optimal_reward,
                "policy_optimization/_result/cpg_5_n_obs_{}.pdf".format(n_obs))
