import pickle
import sys

from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from kernel_hearding import *
from kernel_two_sample_test import kernel_two_sample_test
from utils import *

if __name__ == "__main__":

    seed = 2
    np.random.seed(seed)

    NS = [10, 50, 100, 150, 200]
    try:
        # get an index of a multiplier as an argument
        exp_idx = int(sys.argv[1])
        NS = [NS[exp_idx]]
    except:
        pass

    num_experiments = 1000

    # generate data from the marginal distributions P(X_0) and P(X_1)
    significance_level = 0.01

    for n_sample in NS:

        ns = n_sample
        lin_num_rejects = 0
        rbf_num_rejects = 0
        test_powers = np.zeros(2,)

        for _ in tqdm(range(num_experiments)):
            ### generate data
            X0, X1, Y0, Y1 = gen_data(ns)

            ### calculate the test statistics and p-value
            YY0 = Y0[:, np.newaxis]

            Yt = np.linspace(-4, 4, ns)
            Yt = Yt[:, np.newaxis]
            sigma = np.median(pairwise_distances(YY0, Yt, metric='sqeuclidean'))

            ### estimate the CME and generate sample using kernel herding
            weights = estimate_cme(X0, X1, Y1)
            Y_samples, _ = generate_herding_samples(ns, YY0, sigma, weights)
            Y_samples = np.array(Y_samples)[:, np.newaxis]

            # Gaussian RBF kernel
            mmd2u_rbf, mmd2u_null_rbf, p_value_rbf = kernel_two_sample_test(Y_samples, YY0,
                                                                            kernel_function='rbf',
                                                                            gamma=1.0 / sigma,
                                                                            verbose=False)

            rbf_num_rejects += int(p_value_rbf < significance_level)

            # linear kernel
            mmd2u_lin, mmd2u_null_lin, p_value_lin = kernel_two_sample_test(Y_samples, YY0,
                                                                            kernel_function='linear',
                                                                            verbose=False)

            lin_num_rejects += int(p_value_lin < significance_level)

        # calculate the test powers
        test_powers[0] = lin_num_rejects / num_experiments
        test_powers[1] = rbf_num_rejects / num_experiments

        # save the results
        with open("_exp_results/power_analysis_s{}_n{}.pickle".format(seed, n_sample), 'wb') as handle:
            pickle.dump(test_powers, handle, protocol=pickle.HIGHEST_PROTOCOL)
