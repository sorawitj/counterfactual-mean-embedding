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
from scipy.stats.mstats import winsorize

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
        context_dim = np.shape(sim_data['null_context_vec'][0])[0]
        reco_dim = np.shape(sim_data['null_reco_vec'][0])[0]
        hidden_units = [40]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        numpy_input = {'context_vec': np.stack(sim_data['null_context_vec'].as_matrix()),
                       'reco_vec': np.stack(sim_data['null_reco_vec'].as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, sim_data['null_reward'].as_matrix(),
                                                            batch_size=1024, num_epochs=100, shuffle=True)
        classifier.train(input_fn=train_input_fn)

        numpy_input = {'context_vec': np.stack(sim_data['target_context_vec'].as_matrix()),
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

    def calculate_weight(self, row):
        nullProb = self.null_policy.get_propensity(row.null_multinomial, row.null_reco)
        if not self.target_policy.greedy:
            targetProb = self.target_policy.get_propensity(row.target_multinomial, row.null_reco)
        else:
            targetProb = 1.0 if row.null_reco == row.target_reco else 0

        return targetProb / nullProb

    def estimate(self, sim_data):
        sim_data['ips_w'] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data['ips_w'] * sim_data['null_reward'])
        exp_weight = np.mean(sim_data['ips_w'])

        return exp_reward / exp_weight


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

    def estimate(self, sim_data):
        sim_data = sim_data.copy()
        context_dim = np.shape(sim_data['null_context_vec'][0])[0]
        reco_dim = np.shape(sim_data['null_reco_vec'][0])[0]
        hidden_units = [40]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        null_numpy_input = {'context_vec': np.stack(sim_data['null_context_vec'].as_matrix()),
                            'reco_vec': np.stack(sim_data['null_reco_vec'].as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, sim_data['null_reward'].as_matrix(),
                                                            batch_size=1024, num_epochs=100, shuffle=True)
        classifier.train(input_fn=train_input_fn)

        target_numpy_input = {'context_vec': np.stack(sim_data['target_context_vec'].as_matrix()),
                              'reco_vec': np.stack(sim_data['target_reco_vec'].as_matrix())}

        null_pred_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, batch_size=1024, num_epochs=1,
                                                                shuffle=False)
        target_pred_input_fn = tf.estimator.inputs.numpy_input_fn(target_numpy_input, batch_size=1024, num_epochs=1,
                                                                  shuffle=False)

        null_predictions = []
        target_predictions = []

        null_prediction = classifier.predict(null_pred_input_fn)
        for null_p in null_prediction:
            null_predictions.append(null_p['class_ids'][0])

        target_prediction = classifier.predict(target_pred_input_fn)
        for target_p in target_prediction:
            target_predictions.append(target_p['class_ids'][0])

        sim_data['null_pred'] = null_predictions
        sim_data['target_pred'] = target_predictions
        sim_data['ips_w'] = sim_data.apply(self.calculate_weight, axis=1)
        sim_data['ips_w'] = winsorize(sim_data['ips_w'], (0.0, 0.01))

        estimated_reward = sim_data['target_pred'] + (sim_data['null_reward'] - sim_data['null_pred']) * sim_data['ips_w']

        return np.mean(estimated_reward)


class SlateEstimator(Estimator):
    @property
    def name(self):
        return "slate_estimator"

    def __init__(self, n_reco, null_policy):
        self.n_reco = n_reco
        self.null_policy = null_policy

    def calculate_weight(self, row):
        n_items = self.null_policy.n_items
        n_reco = self.n_reco
        n_dim = n_reco * n_items
        temp_range = range(n_reco)

        exploredMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        exploredMatrix[temp_range, list(row.null_reco)] = 1.

        targetMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        targetMatrix[temp_range, list(row.target_reco)] = 1.

        posRelVector = exploredMatrix.reshape(n_dim)
        targetSlateVector = targetMatrix.reshape(n_dim)

        estimatedPhi = np.dot(self.null_policy.gammas[row.user], posRelVector)

        return np.dot(estimatedPhi, targetSlateVector)

    def estimate(self, sim_data):
        sim_data['ips_w'] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data['ips_w'] * sim_data['null_reward'])
        exp_weight = np.mean(sim_data['ips_w'])

        return exp_reward / exp_weight


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
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def estimate(self, sim_data):
        """
         Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
         """

        # extract the regularization and kernel parameters
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        null_reward = sim_data.null_reward.dropna(axis=0)

        null_context_vec = np.stack(sim_data['null_context_vec'].dropna(axis=0).as_matrix())
        null_reco_vec = np.stack(sim_data['null_reco_vec'].dropna(axis=0).as_matrix())
        target_context_vec = np.stack(sim_data['target_context_vec'].dropna(axis=0).as_matrix())
        target_reco_vec = np.stack(sim_data['target_reco_vec'].dropna(axis=0).as_matrix())

        # use median heuristic for the bandwidth parameters
        context_param = (0.5 * context_param) / np.median(pdist(np.vstack([null_context_vec, target_context_vec]), 'sqeuclidean'))
        recom_param = (0.5 * recom_param) / np.median(pdist(np.vstack([null_reco_vec, target_reco_vec]), 'sqeuclidean'))

        contextMatrix = self.context_kernel(null_context_vec, null_context_vec, context_param)
        recomMatrix = self.recom_kernel(null_reco_vec, null_reco_vec, recom_param) #

        targetContextMatrix = self.context_kernel(null_context_vec, target_context_vec, context_param)
        targetRecomMatrix = self.recom_kernel(null_reco_vec, target_reco_vec, recom_param)

        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["target_reco"].dropna(axis=0).shape[0]
        n = sim_data["null_reco"].dropna(axis=0).shape[0]
        b = np.dot(np.multiply(targetContextMatrix, targetRecomMatrix), np.repeat(1.0 / m, m, axis=0))

        # solve a linear least-square
        A = np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n))
        beta_vec,_ = scipy.sparse.linalg.cg(A, b, tol=1e-06)

        # return the expected reward as an average of the rewards, obtained from the null policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        return np.dot(beta_vec, null_reward)