from abc import abstractmethod

import numpy as np
import itertools
import scipy
import scipy.linalg
import pandas as pd
from Policy import *
from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import pdist
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

import joblib
from sklearn.model_selection import StratifiedKFold

"""
Classes that represent different policy estimators for simulated experiments
"""

class Estimator(object):
    @abstractmethod
    def estimate(self, sim_data):
        """
        calculate expected reward from an observation
        :param sim_data: a data frame consists of {context, null_reco, null_reward, target_reco, target_reward}
        :return: expected reward (double)
        """
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class DirectEstimator(Estimator):
    @property
    def name(self):
        return "direct_estimator"

    def estimate(self, sim_data):
        sim_data = sim_data.copy()
        context_dim = np.shape(sim_data['context_vec'][0])[0]
        reco_dim = np.shape(sim_data['null_reco_vec'][0])[0]
        hidden_units = [40]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        numpy_input = {'context_vec': np.stack(sim_data['context_vec'].as_matrix()),
                       'reco_vec': np.stack(sim_data['null_reco_vec'].as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, sim_data['null_reward'].as_matrix(),
                                                            batch_size=1024, num_epochs=100, shuffle=True)
        classifier.train(input_fn=train_input_fn)

        numpy_input = {'context_vec': np.stack(sim_data['context_vec'].as_matrix()),
                       'reco_vec': np.stack(sim_data['target_reco_vec'].as_matrix())}
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, batch_size=1024, num_epochs=1, shuffle=False)
        prediction = classifier.predict(pred_input_fn)
        total_reward = 0
        n = 0
        for p in prediction:
            total_reward += p['logistic'][0]
            n += 1
        expReward = total_reward / n

        return expReward


class IPSEstimator(Estimator):
    @property
    def name(self):
        return "ips_estimator"

    def __init__(self, n_reco: int, null_policy: MultinomialPolicy, target_policy: MultinomialPolicy):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param target_policy: a policy that we want to estimate its reward
        """
        self.n_reco = n_reco
        self.null_policy = null_policy
        self.target_policy = target_policy

    def single_estimate(self, row):
        nullProb = self.null_policy.get_propensity(row.null_multinomial, row.null_reco)
        targetProb = self.target_policy.get_propensity(row.target_multinomial, row.null_reco)

        return row.null_reward * targetProb / nullProb

    def estimate(self, sim_data):
        expReward = sim_data.apply(self.single_estimate, axis=1)
        # expReward = applyParallel(sim_data, self.single_estimate)
        return np.mean(expReward)


class DoublyRobustEstimator(IPSEstimator):
    @property
    def name(self):
        return "doubly robust estimator"

    def __init__(self, n_reco: int, null_policy: MultinomialPolicy, target_policy: MultinomialPolicy):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param target_policy: a policy that we want to estimate its reward

        """
        super().__init__(n_reco, null_policy, target_policy)

    def single_estimate(self, row):
        nullProb = self.null_policy.get_propensity(row.null_multinomial, row.null_reco)
        targetProb = self.target_policy.get_propensity(row.target_multinomial, row.null_reco)

        estimated_reward = row.target_pred + (row.null_reward - row.null_pred) * targetProb / nullProb

        return estimated_reward

    def estimate(self, sim_data):
        sim_data = sim_data.copy()
        context_dim = np.shape(sim_data['context_vec'][0])[0]
        reco_dim = np.shape(sim_data['null_reco_vec'][0])[0]
        hidden_units = [40]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        null_numpy_input = {'context_vec': np.stack(sim_data['context_vec'].as_matrix()),
                            'reco_vec': np.stack(sim_data['null_reco_vec'].as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, sim_data['null_reward'].as_matrix(),
                                                            batch_size=1024, num_epochs=100, shuffle=True)
        classifier.train(input_fn=train_input_fn)

        target_numpy_input = {'context_vec': np.stack(sim_data['context_vec'].as_matrix()),
                              'reco_vec': np.stack(sim_data['target_reco_vec'].as_matrix())}

        null_pred_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, batch_size=1024, num_epochs=1,
                                                                shuffle=False)
        target_pred_input_fn = tf.estimator.inputs.numpy_input_fn(target_numpy_input, batch_size=1024, num_epochs=1,
                                                                  shuffle=False)

        null_predictions = []
        target_predictions = []

        null_prediction = classifier.predict(null_pred_input_fn)
        for null_p in null_prediction:
            null_predictions.append(null_p['logistic'][0])

        target_prediction = classifier.predict(target_pred_input_fn)
        for target_p in target_prediction:
            target_predictions.append(target_p['logistic'][0])

        sim_data['null_pred'] = null_predictions
        sim_data['target_pred'] = target_predictions
        expReward = sim_data.apply(self.single_estimate, axis=1)
        # expReward = applyParallel(sim_data, self.single_estimate)

        return np.mean(expReward)


class SlateEstimator(Estimator):
    @property
    def name(self):
        return "slate_estimator"

    def __init__(self, n_reco, null_policy):

        self.n_reco = n_reco
        self.null_policy = null_policy

    def gamma_inverse(self, context):
        """
        calculate inverse of gamma
        see Off-policy evaluation for slate recommendation(https://arxiv.org/pdf/1605.04812.pdf) for more details
        :return: gamma matrix inverse
        """

        n_items = self.null_policy.n_items

        n_reco = self.n_reco

        gamma = np.zeros((n_items * n_reco, n_items * n_reco),
                         dtype=np.longdouble)
        for p in range(n_reco):
            currentStart = p * n_items
            currentEnd = p * n_items + n_items
            currentMarginals = np.sum(self.prob_dist[context], axis=tuple([q for q in range(n_reco) if q != p]),
                                      dtype=np.longdouble)
            gamma[currentStart:currentEnd, currentStart:currentEnd] = np.diag(currentMarginals)

        for p in range(n_reco):
            for q in range(p + 1, n_reco):
                currentRowStart = p * n_items
                currentRowEnd = p * n_items + n_items
                currentColumnStart = q * n_items
                currentColumnEnd = q * n_items + n_items
                pairMarginals = np.sum(self.prob_dist[context],
                                       axis=tuple([r for r in range(n_reco) if r != p and r != q]),
                                       dtype=np.longdouble)
                np.fill_diagonal(pairMarginals, 0)

                gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd] = pairMarginals
                gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd] = pairMarginals.T

        return scipy.linalg.pinv(gamma, cond=1e-15, rcond=1e-15)

    def single_estimate(self, context, null_reco, null_reward, target_reco):
        n_items = self.null_policy.n_items
        n_reco = min(n_items, self.n_reco)
        n_dim = n_reco * n_items
        temp_range = range(n_reco)

        exploredMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        exploredMatrix[temp_range, list(null_reco[0:n_reco])] = null_reward

        targetMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        targetMatrix[temp_range, list(target_reco[0:n_reco])] = 1

        posRelVector = exploredMatrix.reshape(n_dim)
        targetSlateVector = targetMatrix.reshape(n_dim)

        estimatedPhi = np.dot(self.gamma_inverse(context), posRelVector)

        return np.dot(estimatedPhi, targetSlateVector)


"""
The counterfactual mean embedding estimator 
"""
class CMEstimator(Estimator):
    @property
    def name(self):
        return "cme_estimator"

    def __init__(self, context_kernel, recom_kernel, params):
        """
         :param context_kernel: the kernel function for the context variable
         :param recom_kernel: the kernel function for the recommendation
         :param params: all parameters including regularization parameter and kernel parameters
         """

        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self.params = params

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, value):
        self.__params = value

    def select_parameters(self, sim_data, params_grid, n_splits=5):
        """
        Select the best parameter setting

        :return: the best parameters
        """

        num_params = len(params_grid)
        num_data = len(sim_data)

        # create estimators using parameter grid
        cme_estimators = [CMEstimator(self.context_kernel, self.recom_kernel, params) for params in params_grid]

        kfold = StratifiedKFold(n_splits=n_splits)

        sq_errors = np.zeros(num_params)

        with joblib.Parallel(n_jobs=num_params, max_nbytes=1e6) as parallel:

            for train, test in kfold.split(np.zeros(num_data), sim_data.null_reward):

                sim_data_null = sim_data.iloc[train]

                # evaluate the estimator on each split
                actual_value = sim_data["null_reward"].iloc[test].mean()
                estimated_values = parallel(joblib.delayed(e.estimate)(sim_data_null) for e in cme_estimators)
                sq_errors += [(est - actual_value) ** 2 for est in estimated_values]

            sq_errors /= n_splits

        # set and return the best parameters
        best_params = params_grid[np.argmin(sq_errors)]
        self.params = best_params

        return best_params

    def estimate(self, sim_data):
        """
         Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
         """

        # extract the regularization and kernel parameters
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        null_reward = sim_data.null_reward
        context_vec = np.stack(sim_data.context_vec.as_matrix())
        null_reco_vec = np.stack(sim_data.null_reco_vec.as_matrix())
        target_reco_vec = np.stack(sim_data.target_reco_vec.as_matrix())

        # use median heuristic for the bandwidth parameters
        context_param = 0.5 / np.median(pdist(context_vec, 'sqeuclidean'))
        null_recom_param = (0.5 * recom_param) / np.median(pdist(null_reco_vec, 'sqeuclidean'))
        target_recom_param = (0.5 * recom_param) / np.median(pdist(target_reco_vec, 'sqeuclidean'))
        # recom_param = 0.5 * (null_recom_param + target_recom_param)

        contextMatrix = self.context_kernel(context_vec, context_vec, context_param)
        targetContextMatrix = self.context_kernel(context_vec, context_vec, context_param)
        recomMatrix = self.recom_kernel(null_reco_vec, null_reco_vec, target_recom_param) #
        targetRecomMatrix = self.recom_kernel(null_reco_vec, target_reco_vec, target_recom_param)

        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["target_reco"].shape[0]
        n = sim_data["null_reco"].shape[0]
        b = np.dot(np.multiply(targetContextMatrix, targetRecomMatrix), np.repeat(1.0 / m, m, axis=0))

        # solve a linear least-square
        A = np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n))
        #beta_vec = np.linalg.solve(A, b)

        #Ainv = scipy.sparse.linalg.splu(A)
        #Ainv = scipy.sparse.linalg.LinearOperator(A.size, Ainv.solve)
        beta_vec,_ = scipy.sparse.linalg.cg(A, b, tol=1e-06, maxiter=1000)

        # return the expected reward as an average of the rewards, obtained from the null policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        return np.dot(beta_vec, null_reward)
