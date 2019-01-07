import argparse
import sys
from functools import partial

import numpy as np
from PolicyGradient import *
from CME import *
from DirectClassification import *
from plot_fn import plot_comparison_result
import multiprocessing
import pandas as pd


def run_cme_reg(reg_pow, train_context, test_context, train_action, train_rewards, test_action, test_rewards):
    estimator = CME(train_context, train_action, train_rewards, reg_pow)
    estiamted_reward = estimator.estimate(test_context, test_action).mean()
    test_reward = test_rewards.mean()
    return np.square(estiamted_reward - test_reward)


def cross_validate_CME(reg_pows, context, null_actions, null_rewards, n_splits=5):
    N = null_actions.shape[0]
    n_params = len(reg_pows)
    # random split corss validation set
    index = np.arange(N)
    np.random.shuffle(index)
    fold_indices = np.array_split(index, n_splits)

    errors = np.zeros(shape=(n_splits, n_params))

    for i in range(n_splits):
        mask = np.ones(N, dtype=bool)
        mask[fold_indices[i]] = False
        train_action, train_rewards, train_context = null_actions[mask], null_rewards[mask], context[mask]
        test_action, test_rewards, test_context = null_actions[fold_indices[i]], null_rewards[fold_indices[i]], context[
            fold_indices[i]]

        func = partial(run_cme_reg,
                       train_context=train_context,
                       test_context=test_context,
                       train_action=train_action,
                       train_rewards=train_rewards,
                       test_action=test_action,
                       test_rewards=test_rewards)

        with multiprocessing.Pool(n_params) as p:
            error = p.map(func, reg_pows)

        errors[i] = np.array(error)

    mean_errors = errors.mean(axis=0)
    min_idx = np.argmin(mean_errors)

    return reg_pows[min_idx], mean_errors[min_idx]


def get_rewards(action, context, true_w):
    true_mean = context.dot(true_w)
    return np.exp(-0.5 * ((true_mean - action) ** 2))


def run_iteration(context,
                  null_w,
                  true_w,
                  null_actions,
                  null_rewards,
                  config,
                  est='CME',
                  reg_pow=1.0):
    # decide which estimator to use
    if est == 'Direct':
        null_feature_vec = np.hstack([context, null_actions[:, np.newaxis]])
        estimator = DirectRegression(null_feature_vec, null_rewards)
    elif est == 'CME':
        estimator = CME(context, null_actions, null_rewards, reg_pow)
    else:
        sys.exit(1)

    target_pred_rewards = []
    target_exp_rewards = []

    policy_grad_graph = tf.Graph()
    sess = tf.Session(graph=policy_grad_graph)
    with policy_grad_graph.as_default():
        policy_grad = PolicyGradientGaussian(config, sess, null_w)
        sess.run(tf.global_variables_initializer())

    learning_rates = np.logspace(np.log2(0.2), np.log2(0.02), config['num_iter'], base=2)

    for i in range(config['num_iter']):

        target_actions, target_action_probs = policy_grad.act(context)

        if est == 'Direct':
            train_actions = target_actions
        elif est == 'CME':
            train_actions = null_actions
        else:
            sys.exit(1)

        target_reward_vec = estimator.estimate(context, target_actions)

        target_reward = target_reward_vec.mean()
        expected_reward = get_rewards(target_actions.flatten(), context, true_w).mean()
        target_pred_rewards.append(target_reward)
        target_exp_rewards.append(expected_reward)

        loss = policy_grad.train_step(context,
                                      train_actions,
                                      target_reward_vec,
                                      learning_rates[i])

        if i % 20 == 0:
            print("iter {}, Expected reward: {}".format(i, expected_reward))
            print("iter {}, Predicted reward: {}".format(i, target_reward))
            print("iter {}, loss: {}".format(i, loss))

    sess.close()

    return target_exp_rewards, target_pred_rewards


if __name__ == "__main__":
    config = {
        "context_dim": 3,
        'num_iter': 600,
        'n_obs': 5000,
        'scale': 1.0
    }

    try:
        # get an index of a multiplier as an argument
        reg_pow_idx = int(sys.argv[1])
    except:
        sys.exit(1)

    rand_state = np.random.RandomState(111)

    p_component = (0.2, 0.3, 0.1, 0.3, 0.1)
    mu_component = rand_state.uniform(-2.0, 2.0, size=6)
    # mu_component = rand_state.uniform(0, 0, size=6)
    sd_component = np.repeat(1.0, 6)

    context = np.zeros(shape=(config['n_obs'], config['context_dim']))
    for i in range(config['n_obs']):
        context_components = rand_state.choice(5, size=1, p=p_component, replace=True)
        context[i] = rand_state.normal(0, 1.0, size=(config['context_dim'])) * sd_component[
            context_components, np.newaxis] + \
                     mu_component[
                         context_components, np.newaxis]

    true_w = rand_state.normal(size=config['context_dim'], scale=config['scale'])

    null_w = rand_state.normal(size=config['context_dim'], scale=config['scale'])
    # null_w = -true_w

    null_mu = context.dot(null_w)
    null_actions = rand_state.normal(null_mu, scale=1.)
    null_rewards = get_rewards(null_actions, context, true_w)

    # reg_pows = [-3, -2, -1, 0, 1, 2, 3]
    reg_pows = [-2, -1.5, -1, 0, 1, 1.5, 2]
    if reg_pow_idx <= 6:
        estimator = 'CME'
        reg_pow = reg_pows[reg_pow_idx]
    else:
        estimator = 'Direct'
        reg_pow = 0

    var_rewards = np.zeros(config['num_iter'])
    exp_reward, pred_reward = \
        run_iteration(context,
                      null_w,
                      true_w,
                      null_actions,
                      null_rewards,
                      config,
                      estimator,
                      reg_pow)

    optimal_actions = rand_state.normal(context.dot(true_w), scale=1.0)
    optimal_rewards = get_rewards(optimal_actions, context, true_w)

    res_df = pd.DataFrame([exp_reward, pred_reward]).T
    res_df.columns = ['exp_reward', 'pred_reward']
    res_df['optimal_reward'] = optimal_rewards.mean()

    dir_path = 'policy_opt_results/complex_direct/'
    if estimator == 'CME':
        file_name = dir_path + 's' + str(config['scale']) + 'n' + str(config['n_obs']) + 'd' + str(
            config['context_dim']) + \
                    estimator + '_' + str(reg_pow) + '.csv'
    else:
        file_name = dir_path + 's' + str(config['scale']) + 'n' + str(config['n_obs']) + 'd' + str(
            config['context_dim']) + \
                    estimator + '.csv'

    res_df.to_csv(file_name, index=False)

    ndim = 3
    n_obs = 5000
    scale = 1.0
    n_iters = 600
    exp_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
    pred_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
    optimal_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
    estimators = []
    dir_path = 'policy_opt_results/complex_direct/'
    for i in range(len(reg_pows)):
        df = pd.read_csv(dir_path + 's{}n{}d{}CME_'.format(scale, n_obs, ndim) + str(reg_pows[i]) + '.csv')
        exp_rewards[i] = df['exp_reward']
        pred_rewards[i] = df['pred_reward']
        optimal_rewards[i] = df['optimal_reward'].mean()
        estimators.append('CME_reg_' + str(reg_pows[i]))

    df = pd.read_csv(dir_path + 's{}n{}d{}Direct.csv'.format(scale, n_obs, ndim))
    exp_rewards[len(reg_pows)] = df['exp_reward']
    pred_rewards[len(reg_pows)] = df['pred_reward']
    optimal_rewards[len(reg_pows)] = df['optimal_reward'].mean()
    estimators.append('Direct')

    var_rewards = np.zeros(n_iters)
    pred_rewards = np.zeros(shape=pred_rewards.shape)

    plot_comparison_result(exp_rewards,
                           pred_rewards,
                           var_rewards,
                           optimal_rewards.mean(),
                           0.0,
                           "policy_optimization/_result/guassian_policy_d{}.pdf".format(ndim),
                           "CME vs Direct: context_dim = {}".format(ndim),
                           estimators,
                           ylim=(0.01, 0.75))
