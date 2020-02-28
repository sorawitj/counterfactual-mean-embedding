import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

main_dir = 'counterfactual_sample/_exp_results'
sample_size = [10, 50, 100, 150, 200]
df_res = pd.DataFrame(columns=['Power', 'Kernel'])
for n_sample in sample_size:
    file = "power_analysis_s{}_n{}.pickle".format(2, n_sample)
    powers = pickle.load(open(os.path.join(main_dir, file), 'rb'))
    df_n = pd.DataFrame({'Kernel': np.array(['ATE', 'DATE']),
                         'Power': powers})
    df_n['Sample Size'] = n_sample
    df_res = df_res.append(df_n, ignore_index=True)

# df_mae = df_mae.query("Quantity == 'Joint Effect'")
g = sns.FacetGrid(df_res, hue='Kernel', height=3.4*0.8, aspect=1.2,
                  sharey=False, hue_kws={"marker": ["o", "s"]})
(g.map(sns.lineplot, 'Sample Size', 'Power', markers=True, dashes=False))

g.ax.legend()
g.ax.set_title("Power of Test")
plt.tight_layout()
plt.savefig('counterfactual_sample/hearding_power_analysis2.pdf')
