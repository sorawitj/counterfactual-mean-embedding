from ParameterSelector import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class Direct(object):

    def __init__(self, null_feature_vec, null_reward, params=(40, 1024, 100)):
        self.null_feature_vec = null_feature_vec
        self.null_reward = null_reward
        self.params = params

        # train a classifier
        dim = self.null_feature_vec.shape[1]
        hidden_units = [self.params[0]]
        feature_columns = [tf.feature_column.numeric_column('feature_vec', shape=(dim,))]
        self.classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                feature_columns=feature_columns,
                                                optimizer='Adam',
                                                dropout=0.2)

        numpy_input = {'feature_vec': np.stack(self.null_feature_vec)}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, self.null_reward,
                                                            batch_size=self.params[1], num_epochs=self.params[2],
                                                            shuffle=True)
        self.classifier.train(input_fn=train_input_fn)
        
    def estimate(self, target_feature_vec):

        numpy_input = {'feature_vec': np.stack(target_feature_vec)}
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(numpy_input, batch_size=self.params[1], num_epochs=1,
                                                           shuffle=False)
        prediction = self.classifier.predict(pred_input_fn)
        total_reward = 0
        n = 0
        for p in prediction:
            total_reward += p['logistic'][0]
            n += 1
        expReward = total_reward / n

        return expReward
