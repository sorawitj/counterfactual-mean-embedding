import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize

def winsorized_std(data, percentile):
    std = np.std(winsorize(data, (0, percentile)).data)
    return std

def winsorized_mean(data, percentile):
    mean = np.mean(winsorize(data, (0, percentile)).data)
    return mean


prelim_result = pd.read_csv("prelim_result.csv")


prelim_result = prelim_result[[c for c in prelim_result.columns if 'error' in c] + ['multiplier']]
# prelim_result = prelim_result.query("multiplier < -0.2")
df = pd.melt(prelim_result, id_vars=["multiplier"], var_name="estimator")

ax = df.groupby(["estimator", "multiplier"]).agg(lambda x: winsorized_mean(x, 0.1)).unstack("estimator")['value'].plot(ylim=(0,2),logy=True)

x = df.multiplier.unique()
palette = sns.color_palette()

for cond, cond_df in df.groupby("estimator"):
    sd = cond_df.groupby("multiplier").value.apply(winsorized_std, 0.1)
    mean = cond_df.groupby("multiplier").value.apply(winsorized_mean, 0.1)
    n = cond_df.groupby("multiplier").size()
    low = mean + sd/n
    high = mean - sd/n
    ax.fill_between(x, low, high, alpha=.2, color=palette.pop(0))