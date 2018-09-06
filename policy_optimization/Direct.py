from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class Direct(object):

    def __init__(self, null_feature_vec, null_rewards):
        self.params = params

    def estimate(self, data):
        data = data.copy()

        null_context_vec = data['null_context_vec'].dropna(axis=0)
        null_reco_vec = data['null_reco_vec'].dropna(axis=0)
        target_context_vec = data['target_context_vec'].dropna(axis=0)
        target_reco_vec = data['target_reco_vec'].dropna(axis=0)
        null_reward = data['null_reward'].dropna(axis=0)

        context_dim = null_context_vec.iloc[0].shape[0]
        reco_dim = null_reco_vec.iloc[0].shape[0]

        hidden_units = [self.params[0]]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        numpy_input = {'context_vec': np.stack(null_context_vec.as_matrix()),
                       'reco_vec': np.stack(null_reco_vec.as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, null_reward.as_matrix(),
                                                            batch_size=self.params[1], num_epochs=self.params[2],
                                                            shuffle=True)
        classifier.train(input_fn=train_input_fn)

        numpy_input = {'context_vec': np.stack(target_context_vec.as_matrix()),
                       'reco_vec': np.stack(target_reco_vec.as_matrix())}
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, batch_size=self.params[1], num_epochs=1,
                                                           shuffle=False)
        prediction = classifier.predict(pred_input_fn)
        total_reward = 0
        n = 0
        for p in prediction:
            total_reward += p['logistic'][0]
            n += 1
        expReward = total_reward / n

        return expReward
