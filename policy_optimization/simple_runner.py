import sys
sys.path.append("../policy_evaluation/")

from CME import *
from ParameterSelector import *
from PolicyGradient import *
import numpy as np
import matplotlib.pyplot as plt

def get_sample_rewards(weight_vector, actions):
    return np.random.binomial(1, p=sigmoid(
        sample_users[np.arange(len(sample_users)), actions, :].dot(weight_vector.T)))


def get_expected_reward(weight_vector, actions):
    p = sigmoid(sample_users[np.arange(len(sample_users)), actions, :].dot(weight_vector.T))
    return p.mean()

# main
if __name__=="__main__":

    config = {
        "n_users": 10,
        "n_items": 5,
        "n_observation": 3000,
        "context_dim": 10,
        'learning_rate': 0.05
    }

    SHOW_PLOT = True
    num_iters = 1000

    user_item_vectors = np.random.normal(0, 1, size=(config['n_users'], config['n_items'], config['context_dim']))

    sample_users = user_item_vectors[np.random.choice(user_item_vectors.shape[0], config['n_observation'], True), :]

    print(np.shape(sample_users))

    true_weights = np.random.normal(0, 1, config['context_dim'])
    null_policy_weight = np.random.normal(0, 1, config['context_dim'])
    null_action_probs = softmax(sample_users.dot(null_policy_weight.T), axis=1)
    null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))
    null_rewards = get_sample_rewards(true_weights, null_actions)

    sess = tf.Session()
    policy_grad = PolicyGradientAgent(config, sess)
    sess.run(tf.global_variables_initializer())

    null_feature_vec = sample_users[np.arange(len(sample_users)), null_actions, :]
    cmeEstimator = CME(null_feature_vec, null_rewards)

    loss_trace = np.zeros(num_iters)
    
    print("+--------------------------------------------------------------------------------------------------------+")
    print("iter \t\t Exp Reward \t\t\t CME Reward \t\t\t Loss")
    print("+--------------------------------------------------------------------------------------------------------+")
    for i in range(num_iters):
        
        target_actions = policy_grad.act(sample_users)
        target_feature_vec = sample_users[np.arange(len(sample_users)), target_actions, :]
        target_reward = cmeEstimator.estimate(target_feature_vec)

        loss = policy_grad.train_step(sample_users, null_actions, target_reward)
        loss_trace[i] = loss

        if i % 20 == 0:
            print("{} \t\t {} \t\t {} \t\t {}".format(i, get_expected_reward(true_weights, target_actions),target_reward.sum(),loss))

    print("+--------------------------------------------------------------------------------------------------------+")

    target_reward = get_expected_reward(true_weights, target_actions)

    optimal_actions = np.argmax(sample_users.dot(true_weights.T), axis=1)
    optimal_rewards = get_expected_reward(true_weights, optimal_actions)

    print("Target policy: Expected reward: {}".format(target_reward))
    print("Optimal policy: Expected reward: {}".format(optimal_rewards))

    if SHOW_PLOT:
        # plot the trace
        plt.plot(loss_trace)
        plt.show()
