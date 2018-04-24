import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize


def winsorized_std(data, percentile):
    std = np.std(winsorize(data, (0, percentile)).data)
    return std


def winsorized_mean(data, percentile):
    mean = np.mean(winsorize(data, (0, percentile)).data)
    return mean


prelim_result = pd.read_csv("prelim_result2.csv")

prelim_result = prelim_result[[c for c in prelim_result.columns if 'error' in c] + ['multiplier']]
prelim_result.columns = prelim_result.columns.str.replace("_square_error", "")

estimator_cols = list(filter(lambda x: 'estimator' in x, prelim_result))

winsorized_df = pd.DataFrame()
for cond, cond_df in prelim_result.groupby("multiplier"):
    for e in estimator_cols:
        cond_df[e] = winsorize(cond_df[e], (0.0, 0.0))

    cond_df["multiplier"] = cond
    winsorized_df = winsorized_df.append(cond_df)

# prelim_result = prelim_result.query("multiplier < -0.2")
final_df = pd.melt(winsorized_df, id_vars=["multiplier"], var_name="estimator", value_name="MSE")

ax = sns.pointplot(x="multiplier", y="MSE", hue="estimator", data=final_df)
ax.set_yscale('log')
ax.set_ylabel("Mean Square Error (log scale)")

plt.savefig('prelim_result2.png')