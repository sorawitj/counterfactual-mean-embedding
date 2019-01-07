import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import pandas as pd


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
