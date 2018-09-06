from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class DR(object):

    def __init__(self, n_reco, null_policy, target_policy, params=(40, 1024, 100)):
        """
        :param n_reco: number of recommendation
        :param null_policy: a policy used to generate data
        :param target_policy: a policy that we want to estimate its reward

        """
        super().__init__(n_reco, null_policy, target_policy)
        self.params = params

    def estimate(self, data):
        data = data.copy()

        null_context_vec = data['null_context_vec'].dropna(axis=0)
        null_reco_vec = data['null_reco_vec'].dropna(axis=0)
        null_reward = data['null_reward'].dropna(axis=0)
        target_context_vec = data['target_context_vec'].dropna(axis=0)
        target_reco_vec = data['target_reco_vec'].dropna(axis=0)

        context_dim = null_context_vec.iloc[0].shape[0]
        reco_dim = null_reco_vec.iloc[0].shape[0]

        hidden_units = [self.params[0]]
        feature_columns = [tf.feature_column.numeric_column('context_vec', shape=(context_dim,)),
                           tf.feature_column.numeric_column('reco_vec', shape=(reco_dim,))]
        classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        null_numpy_input = {'context_vec': np.stack(null_context_vec.as_matrix()),
                            'reco_vec': np.stack(null_reco_vec.as_matrix())}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, null_reward.as_matrix(),
                                                            batch_size=self.params[1], num_epochs=self.params[2],
                                                            shuffle=True)
        classifier.train(input_fn=train_input_fn)

        target_numpy_input = {'context_vec': np.stack(target_context_vec.as_matrix()),
                              'reco_vec': np.stack(target_reco_vec.as_matrix())}

        null_pred_input_fn = tf.estimator.inputs.numpy_input_fn(null_numpy_input, batch_size=self.params[1],
                                                                num_epochs=1,
                                                                shuffle=False)
        target_pred_input_fn = tf.estimator.inputs.numpy_input_fn(target_numpy_input, batch_size=self.params[1],
                                                                  num_epochs=1,
                                                                  shuffle=False)

        null_predictions = []
        target_predictions = []

        null_prediction = classifier.predict(null_pred_input_fn)
        for null_p in null_prediction:
            null_predictions.append(null_p['class_ids'][0])

        target_prediction = classifier.predict(target_pred_input_fn)
        for target_p in target_prediction:
            target_predictions.append(target_p['class_ids'][0])

        ips_w = winsorize(sim_data.apply(self.calculate_weight, axis=1), (0.0, 0.01))
        estimated_reward = target_predictions + (null_reward - null_predictions) * ips_w

        return np.mean(estimated_reward)
