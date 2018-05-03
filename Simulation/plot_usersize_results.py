import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os

from scipy.stats.mstats import winsorize

plt.switch_backend('agg')

def winsorized_std(data, percentile):
    std = np.std(winsorize(data, (0, percentile)).data)
    return std

def winsorized_mean(data, percentile):
    mean = np.mean(winsorize(data, (0, percentile)).data)
    return mean

# combine the result files
df_list = [pd.read_csv(filename) for filename in glob.glob(os.path.join('jobs/user_size_report/results','*.csv'))]

prelim_result = pd.concat(df_list)
prelim_result.sort_values(by=['num_users'], ascending=True)

prelim_result = prelim_result[[c for c in prelim_result.columns if 'error' in c] + ['num_users']]
prelim_result.columns = prelim_result.columns.str.replace("_square_error", "")

estimator_cols = list(filter(lambda x: 'estimator' in x, prelim_result))

# compute the statistics and plot the results
winsorized_df = pd.DataFrame()
for cond, cond_df in prelim_result.groupby("num_users"):
    for e in estimator_cols:
        cond_df[e] = winsorize(cond_df[e], (0, 0.1))

    cond_df["num_users"] = cond
    winsorized_df = winsorized_df.append(cond_df)

# prelim_result = prelim_result.query("multiplier < -0.2")
final_df = pd.melt(winsorized_df, id_vars=["num_users"], var_name="estimator", value_name="MSE")

# plot the results
ax = sns.pointplot(x="num_users", y="MSE", hue="estimator", data=final_df,
                   linestyles=["-", "--","-.",":",":"], markers=['o','v','x','^','+'])

ax.set_yscale('log')
ax.set_ylabel("Mean Square Error (MSE)")
ax.set_xlabel('Number of Users')

xticks = ['%0.1f' % x for x in np.unique(final_df['num_users'])]
ax.xaxis.set_major_locator(plt.FixedLocator(range(len(xticks))))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(xticks))

leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['CME', 'Direct', 'DR', 'Slate', 'IPS'], title='Estimator')

plt.tight_layout()

plt.savefig('plots/user_size_result.pdf')