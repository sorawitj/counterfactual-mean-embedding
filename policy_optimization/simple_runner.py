import sys
sys.path.append("../policy_evaluation/")

from CME import *
from ParameterSelector import *
from PolicyGradient import *
import numpy as np
import matplotlib.pyplot as plt

config = {
    "n_users": 10,
    "n_items": 5,
    "n_observation": 2000,
    "context_dim": 10,
    'learning_rate': 0.01
}


def get_sample_rewards(weight_vector, actions):
    return np.random.binomial(1, p=sigmoid(
        sample_users[np.arange(len(sample_users)), actions, :].dot(weight_vector.T)))


def get_expected_var_reward(weight_vector, actions):
    p = sigmoid(sample_users[np.arange(len(sample_users)), actions, :].dot(weight_vector.T))
    return p.mean(), (p * (1 - p)).sum()/len(p)**2


np.random.seed(321)

user_item_vectors = np.random.normal(0, 1, size=(config['n_users'], config['n_items'], config['context_dim']))

sample_users = user_item_vectors[np.random.choice(user_item_vectors.shape[0], config['n_observation'], True), :]

true_weights = np.random.normal(0, 1, config['context_dim'])

max_rewards = []
for j in range(100):
    null_policy_weight = np.random.normal(0, 1, config['context_dim'])

    null_action_probs = softmax(sample_users.dot(null_policy_weight.T), axis=1)

    null_actions = np.array(list(map(lambda x: np.random.choice(a=len(x), p=x), null_action_probs)))

    null_rewards = get_sample_rewards(true_weights, null_actions)

    sess = tf.Session()
    policy_grad = PolicyGradientAgent(config, sess, null_policy_weight)
    sess.run(tf.global_variables_initializer())

    null_feature_vec = sample_users[np.arange(len(sample_users)), null_actions, :]
    cme_estimator = CME(null_feature_vec, null_rewards)

    target_cme_rewards = []
    target_exp_rewards = []
    target_var_rewards = []

    for i in range(100):
        
        target_actions = policy_grad.act(sample_users)

        target_feature_vec = sample_users[np.arange(len(sample_users)), target_actions, :]

        target_reward_vec = cme_estimator.estimate(target_feature_vec)
        target_reward = target_reward_vec.sum()
        expected_reward, var_reward = get_expected_var_reward(true_weights, target_actions)
        target_cme_rewards.append(target_reward)
        target_exp_rewards.append(expected_reward)
        target_var_rewards.append(var_reward)

        loss = policy_grad.train_step(sample_users, null_actions, target_reward_vec)

        if i % 20 == 0:
            print("iter {}, Expected reward: {}".format(i, expected_reward))
            print("iter {}, CME reward: {}".format(i, target_reward))
            print("iter {}, loss: {}".format(i, loss))

    max_reward = target_exp_rewards[np.argmax(target_cme_rewards)]
    max_rewards.append(max_reward)

print("FINISH TRAINING !!!")
target_reward, _ = get_expected_var_reward(true_weights, target_actions)

optimal_actions = np.argmax(sample_users.dot(true_weights.T), axis=1)
optimal_reward, _ = get_expected_var_reward(true_weights, optimal_actions)

print("Target policy: Expecrted reward: {}".format(target_reward))
print("Optimal policy: Expected reward: {}".format(optimal_reward))

pd.DataFrame(optimal_reward-max_rewards).hist(bins=10)
plt.show()

# ----------------------- PLOTTING ---------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(mean, color_mean)


# generate 3 sets of random means and confidence intervals to plot
mean0 = np.array(target_exp_rewards)
ub0 = mean0 + 2 * np.array(np.sqrt(target_var_rewards))
lb0 = mean0 - 2 * np.array(np.sqrt(target_var_rewards))

mean1 = np.array(target_cme_rewards)
mean2 = np.repeat(optimal_reward, len(target_cme_rewards))

# plot the data
fig = plt.figure(1, figsize=(8, 3.0))
plot_mean_and_CI(mean0, ub0, lb0, color_mean='b', color_shading='b')
plt.plot(mean1, 'g')
plt.plot(mean2, 'r--')


class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2 * width / 5, y0], width / 5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch

bg = np.array([1, 1, 1])  # background of the legend is white
colors = ['blue', 'green', 'red']
# with alpha = .5, the faded color is the average of the background and color
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]

plt.legend([0, 1, 2], ['Expected Reward', 'CME Reward', 'Optimal Reward'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0]),
               1: LegendObject(colors[1], colors_faded[1]),
               2: LegendObject(colors[2], colors_faded[2], dashed=True),
           })

plt.title('Counterfactual Policy Gradient')
plt.xlabel('number of iterations')
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig('cpg_exp.pdf')
