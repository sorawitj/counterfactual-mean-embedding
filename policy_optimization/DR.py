from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class DR(object):

    def __init__(self, null_feature_vec, null_propensity, null_reward, params=(40, 1024, 100)):
        self.null_feature_vec = null_feature_vec
        self.null_propensity = null_propensity
        self.null_reward = null_reward
        self.params = params

        dim = null_feature_vec.shape[1]

        hidden_units = [self.params[0]]
        feature_columns = [tf.feature_column.numeric_column('feature_vec', shape=(dim,))]
        self.classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        null_numpy_input = {'feature_vec': np.stack(self.null_feature_vec)}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, null_reward,
                                                            batch_size=self.params[1], num_epochs=self.params[2],
                                                            shuffle=True)
        self.classifier.train(input_fn=train_input_fn)

        
    def estimate(self, target_feature_vec, target_propensity):

        null_numpy_input = {'feature_vec': np.stack(self.null_feature_vec)}
        target_numpy_input = {'feature_vec': np.stack(target_feature_vec)}

        null_pred_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, batch_size=self.params[1],
                                                                num_epochs=1,
                                                                shuffle=False)
        target_pred_input_fn = tf.estimator.inputs.numpy_input_fn(target_numpy_input, batch_size=self.params[1],
                                                                  num_epochs=1,
                                                                  shuffle=False)

        null_predictions = []
        target_predictions = []

        null_prediction = self.classifier.predict(null_pred_input_fn)
        for null_p in null_prediction:
            null_predictions.append(null_p['class_ids'][0])

        target_prediction = self.classifier.predict(target_pred_input_fn)
        for target_p in target_prediction:
            target_predictions.append(target_p['class_ids'][0])

        ips_weight = target_propensity / self.null_propensity
        estimated_reward = target_predictions + (self.null_reward - null_predictions) * ips_weight

        return estimated_reward
