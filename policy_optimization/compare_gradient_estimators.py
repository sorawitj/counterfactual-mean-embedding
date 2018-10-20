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
    "n_users": 200,
    "n_items": 20,
    "context_dim": 10,
    'learning_rate': 0.01
}


def get_greedy_reward(item_vectors, actions, sample_users):
    action_vec = item_vectors[actions]
    rewards = np.random.binomial(1, p=sigmoid(
        .5 * (np.sum(sample_users[:, :-1] * action_vec, axis=1) +
              sample_users[:, -1])))
    return rewards


def get_expected_var_reward(item_vectors, action_probs, sample_users):
    p = sigmoid(.5 * (sample_users[:, :-1].dot(item_vectors.T) +
                      sample_users[:, -1, np.newaxis]))
    sum_prob = np.sum(p * action_probs, axis=1)
    # using law of total variance
    evpv = np.sum((p * (1 - p)) * action_probs, axis=1)
    vhm = np.sum(action_probs * (p ** 2), axis=1) - sum_prob ** 2

    return sum_prob.mean(), (evpv + vhm).mean()


def compute_baseline(sample_users, item_vectors, null_actions, null_rewards, config):
    null_action_vec = item_vectors[null_actions]
    ## Baseline Direct Value estimation
    baseline_estimator = Direct(np.hstack([sample_users, null_action_vec]), null_rewards)
    baseline_rewards = np.zeros(shape=(sample_users.shape[0], item_vectors.shape[0]))
    for action in range(config['n_items']):
        target_action_vec = item_vectors[np.repeat(action, sample_users.shape[0])]
        target_feature_vec = np.hstack([sample_users, target_action_vec])
        baseline_rewards[:, action] = baseline_estimator.estimate(target_feature_vec)
    baseline_actions = np.argmax(baseline_rewards, axis=1)
    baseline_action_probs = np.zeros((baseline_actions.shape[0], config['n_items']))
    baseline_action_probs[np.arange(baseline_actions.shape[0]), baseline_actions] = 1
    baseline_reward, baseline_var = get_expected_var_reward(item_vectors, baseline_action_probs, sample_users)
    print("Baseline reward: {}".format(baseline_reward))
    return baseline_reward


def run_iteration(sample_users, item_vectors, null_policy_weight, null_actions, null_rewards, n_observation, num_iter,
                  est='CME'):
    null_action_vec = null_policy_weight[null_actions]

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
        target_action_vec = null_policy_weight[target_actions]

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
        expected_reward = get_greedy_reward(item_vectors, target_actions, sample_users).mean()
        target_pred_rewards.append(target_reward)
        target_exp_rewards.append(expected_reward)
        target_var_rewards.append(0.0)

        loss = policy_grad.train_step(sample_users, train_actions, target_reward_vec)

        if i % 20 == 0:
            print("iter {}, Expected reward: {}".format(i, expected_reward))
            print("iter {}, Predicted reward: {}".format(i, target_reward))
            print("iter {}, loss: {}".format(i, loss))

    sess.close()

    return target_exp_rewards, target_var_rewards, target_pred_rewards


### SIMULATION STARTS HERE ###

np.random.seed(123)

user_components = np.random.choice(5, size=config['n_users'], p=(0.3, 0.1, 0.3, 0.1, 0.2), replace=True)
item_components = np.random.choice(5, size=config['n_items'], p=(0.3, 0.1, 0.3, 0.1, 0.2), replace=True)

# mu_users = np.array([1, -1, 1, -0.5, 0])
mu_users = np.random.uniform(-1.0, 1.0, size=5)
# sd_users = np.array([1, 1.5, 0.5, 0.2, 2])
sd_users = np.random.uniform(0.25, 1.5, size=5)
# mu_items = np.array([0, 1, -0.5, 1, -1])
mu_items = np.random.uniform(-1.0, 1.0, size=5)
sd_items = np.random.uniform(0.25, 1.5, size=5)
# sd_items = np.array([1, 0.4, 1, 1, 0.5])

user_vectors = np.random.normal(0, 1.0, size=(config['n_users'], config['context_dim'] + 1)) \
               * sd_users[user_components, np.newaxis] + mu_users[user_components, np.newaxis]
item_vectors = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim'])) \
               * sd_items[item_components, np.newaxis] + mu_items[item_components, np.newaxis]

# item_vectors[:, -1] = 0.0

# user_components = np.random.choice(5, size=config['n_users'], p=(0.3, 0.1, 0.3, 0.1, 0.2), replace=True)
item_components = np.random.choice(5, size=config['n_items'], p=(0.1, 0.5, 0.2, 0.1, 0.1), replace=True)
# create random null policy
mu_items = np.array([-1, 1, 1, -1, -1.5])
# mu_items = -2*np.abs(mu_items)
sd_items = np.random.uniform(0.25, 1.25, size=5)
null_policy_weight = np.random.normal(0, 1.0, size=(config['n_items'], config['context_dim'])) \
                     * sd_items[item_components, np.newaxis] + mu_items[item_components, np.newaxis]

# create null policy which is different from the optimal policy
# null_policy_weight = -.3 * item_vectors

num_iter = 300
estimators = ['CME', 'Direct']
exp_rewards = np.zeros((len(estimators), num_iter))
pred_rewards = np.zeros((len(estimators), num_iter))
var_rewards = np.zeros((len(estimators), num_iter))

def check_action_dist(actions):
    return pd.DataFrame(actions, columns=['action']).groupby('action').size().reset_index(name='c_act')

for n_obs in [5000]:
    sample_users = user_vectors[np.random.choice(user_vectors.shape[0], n_obs, True), :]
    null_action_probs = softmax(sample_users[:, :-1].dot(null_policy_weight.T) + \
                                sample_users[:, -1, np.newaxis], axis=1)
    null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))
    null_action_dist = check_action_dist(null_actions)
    null_rewards = get_greedy_reward(item_vectors, null_actions, sample_users)

    optimal_actions = np.argmax(sample_users[:, :-1].dot(item_vectors.T) +
                                sample_users[:, -1, np.newaxis], axis=1)
    optimal_action_dist = check_action_dist(optimal_actions)
    optimal_rewards = get_greedy_reward(item_vectors, optimal_actions, sample_users)

    baseline_reward = compute_baseline(sample_users, item_vectors, null_actions, null_rewards, config)

    action_dist = pd.merge(null_action_dist, optimal_action_dist, how='outer', on='action').fillna(0)
    action_dist['c_act_x'] = action_dist['c_act_x']/action_dist['c_act_x'].sum()
    action_dist['c_act_y'] = action_dist['c_act_y']/action_dist['c_act_y'].sum()

    corr = np.corrcoef(item_vectors)

    for i in range(len(estimators)):
        exp_rewards[i], var_rewards[i], pred_rewards[i] = \
            run_iteration(sample_users,
                          item_vectors,
                          null_policy_weight,
                          null_actions,
                          null_rewards,
                          n_obs,
                          num_iter,
                          estimators[i])

    plot_comparison_result(exp_rewards,
                           pred_rewards,
                           var_rewards,
                           optimal_rewards.mean(),
                           baseline_reward,
                           "policy_optimization/_result/compare_est_n_obs_{}.pdf".format(n_obs),
                           "Comparison",
                           estimators)
