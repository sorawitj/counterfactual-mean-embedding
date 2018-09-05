import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(mean, color_mean)


def plot_result(target_exp_rewards,
                target_cme_rewards,
                target_var_rewards,
                optimal_reward,
                save_name):
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
    plt.ylim(0.2, 1.0)
    plt.tight_layout()
    plt.grid()
    plt.show()
    fig.savefig(save_name)
