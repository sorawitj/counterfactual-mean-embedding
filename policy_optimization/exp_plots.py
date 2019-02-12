import numpy as np
import pandas as pd

reg_pows = [-10, -8, -6, -4, -2, 0, 2]
from plot_fn import plot_comparison_result

ndim = 2
n_obs = 3000
scale = 1.0
n_iters = 500
exp_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
pred_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
optimal_rewards = np.zeros(shape=(len(reg_pows) + 1, n_iters))
estimators = []
dir_path = '_policy_opt_results/regularization/'
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
                       "policy_optimization/_result/guassian_policy_d{}s{}.pdf".format(ndim, scale),
                       "CME vs Direct: context_dim = {}, scale ={}".format(ndim, scale),
                       estimators,
                       ylim=(0.5, 0.8))

ndim = 2
n_obs = 3000
n_iters = 500
scales = [0.01, 0.1]

exp_rewards = []
pred_rewards = []
optimal_rewards = []
estimators = []
dir_path = '_policy_opt_results/direct_cme/'

for i in range(len(scales)):
    df = pd.read_csv(dir_path + 's{}n{}d{}CME_-10'.format(scales[i], n_obs, ndim) + '.csv')
    exp_rewards.append(df['exp_reward'])
    pred_rewards.append(df['pred_reward'])
    optimal_rewards.append(df['optimal_reward'].mean())
    estimators.append('scale_{}-CME'.format(str(scales[i])))

for i in range(len(scales)):
    df = pd.read_csv(dir_path + 's{}n{}d{}Direct'.format(scales[i], n_obs, ndim) + '.csv')
    exp_rewards.append(df['exp_reward'])
    pred_rewards.append(df['pred_reward'])
    optimal_rewards.append(df['optimal_reward'].mean())
    estimators.append('scale_{}-Direct'.format(str(scales[i])))

# df = pd.read_csv(dir_path + 's{}n{}d{}Direct.csv'.format(scale, n_obs, ndim))
# exp_rewards[len(scales)] = df['exp_reward']
# pred_rewards[len(scales)] = df['pred_reward']
# optimal_rewards[len(scales)] = df['optimal_reward'].mean()
# estimators.append('Direct')

var_rewards = np.zeros(n_iters)
exp_rewards = np.array(exp_rewards)
pred_rewards = np.array(pred_rewards)
optimal_rewards = np.array(optimal_rewards)
# pred_rewards = np.zeros(shape=pred_rewards.shape)

plot_comparison_result(exp_rewards,
                       pred_rewards,
                       var_rewards,
                       optimal_rewards.mean(),
                       0.0,
                       "policy_optimization/_result/cme_direct_scale_d{}.pdf".format(ndim),
                       "CME(reg = -10) vs Direct: context_dim = {}".format(ndim),
                       estimators,
                       ylim=(0.3, 1.0))
