import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import pandas as pd

from ParameterSelector import ParameterSelector


def softmax(X, tau=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(tau)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def applyParallel(df: pd.DataFrame, func):
    retLst = Parallel(n_jobs=-2)(delayed(func)(row) for i, row in df.iterrows())
    return retLst


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


"""
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.
    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir), my_input_fn)
    This version supports tf 1.4 and above and can be used by pre-made Estimators like tf.estimator.DNNClassifier. 
    Author: Marc Stogaitis
 """


class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            for x in self.next_features:
                yield x

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature)
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError(
                "All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(
                    len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")


def example_input_fn(generator):
    """ An example input function to pass to predict. It must take a generator as input """

    def _inner_input_fn():
        dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.float32)).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {'x': features}

    return _inner_input_fn


def simulate_data(null_policy, target_policy, environment, item_vectors):
    """
    simulate data given policy, environment and set of context
    :return: observations
    """
    user = environment.get_context()
    null_reco, null_multinomial, null_user_vector = null_policy.recommend(user)
    # recommendation is represented by a concatenation of recommended item vectors
    # null_reco_vec = np.mean(item_vectors[null_reco], axis=0)
    null_reco_vec = np.concatenate(item_vectors[null_reco])
    null_reward = environment.get_reward(user, null_reco)

    target_reco, target_multinomial, _ = target_policy.recommend(user)
    # recommendation is represented by a concatenation of recommended item vectors
    # target_reco_vec = np.mean(item_vectors[target_reco], axis=0)
    target_reco_vec = np.concatenate(item_vectors[target_reco])
    target_reward = environment.get_reward(user, target_reco)

    observation = {"null_context_vec": null_user_vector, "target_context_vec": null_user_vector,
                   "null_reco": tuple(null_reco),
                   "null_reco_vec": null_reco_vec, "null_reward": null_reward,
                   "target_reco": tuple(target_reco), "null_multinomial": null_multinomial,
                   "target_multinomial": target_multinomial, "target_reco_vec": target_reco_vec,
                   "target_reward": target_reward, "user": user}

    return observation


def get_actual_reward(target_policy, environment, n=100000):
    sum_reward = 0
    for i in range(n):
        user = environment.get_context()
        target_reco, target_multinomial, _ = target_policy.recommend(user)
        sum_reward += environment.get_reward(user, target_reco)

    return sum_reward / float(n)


def compare_kernel_regression(estimators, null_policy, target_policy, environment, item_vectors, config, seed):
    np.random.seed(seed)
    sim_data = [simulate_data(null_policy, target_policy, environment, item_vectors)
                for _ in range(config['n_observation'])]
    sim_data = pd.DataFrame(sim_data)

    # parameter selection
    direct_selector = ParameterSelector(estimators[0])  # direct estimator
    params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
    direct_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[0] = direct_selector.estimator

    direct_selector = ParameterSelector(estimators[1])  # direct estimator
    params_grid = [0.001, .01, .1, 1, 10]
    direct_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[1] = direct_selector.estimator

    cme_selector = ParameterSelector(estimators[2])  # cme estimator
    params_grid = [[(10.0 ** p) / config['n_observation'], 1.0, 1.0] for p in np.arange(-6, 0, 1)]
    cme_selector.select_from_propensity(sim_data, params_grid, null_policy, target_policy)
    estimators[2] = cme_selector.estimator

    actual_value = get_actual_reward(target_policy, environment)

    estimated_values = dict([(e.name, e.estimate(sim_data)) for e in estimators])
    estimated_values['actual_value'] = actual_value
    estimated_values['null_reward'] = sim_data.null_reward.mean()

    for e in estimators:
        estimated_values[e.name + '_square_error'] = \
            (estimated_values[e.name] - estimated_values['actual_value']) ** 2
    print(estimated_values)

    return estimated_values
