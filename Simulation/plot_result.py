import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import trim_mean

prelim_result = pd.read_csv("prelim_result.csv")


prelim_result = prelim_result[[c for c in prelim_result.columns if 'error' in c] + ['multiplier']]
# prelim_result = prelim_result.query("multiplier < -0.2")
df = pd.melt(prelim_result, id_vars=["multiplier"], var_name="condition")

ax = df.groupby(["condition", "multiplier"]).agg(lambda x: trim_mean(x, 0.1)).unstack("condition").plot(ylim=(0,1),logy=True)

x = df.multiplier.unique()
palette = sns.color_palette()

def trimmed_std(data, percentile):
    data = np.array(data)
    data.sort()
    percentile = percentile / 2.
    low = int(percentile * len(data))
    high = int((1. - percentile) * len(data))
    return data[low:high].std(ddof=0)

for cond, cond_df in df.groupby("condition"):
    sd = cond_df.groupby("multiplier").value.apply(trimmed_std, 0.1)
    mean = cond_df.groupby("multiplier").value.apply(trim_mean, 0.1)
    n = cond_df.groupby("multiplier").size()
    low = mean + 2*sd/n
    high = mean - 2*sd/n
    ax.fill_between(x, low, high, alpha=.2, color=palette.pop(0))