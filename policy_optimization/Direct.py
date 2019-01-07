import numpy as np
import tensorflow as tf

from Utils import FastPredict


class Direct(object):

    def __init__(self, null_feature_vec, null_reward, params=(40, 1024, 100)):
        self.null_feature_vec = null_feature_vec
        self.null_reward = null_reward
        self.params = params
        self.batch_size = len(null_feature_vec)

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

        def predict_input_fn(generator):
            """ An example input function to pass to predict. It must take a generator as input """

            def _inner_input_fn():
                dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.float32)).batch(self.batch_size)
                iterator = dataset.make_one_shot_iterator()
                features = iterator.get_next()
                return {'feature_vec': features}

            return _inner_input_fn

        self.fast_predict = FastPredict(self.classifier, predict_input_fn)

    def estimate(self, target_feature_vec):
        prediction = self.fast_predict.predict(target_feature_vec)

        return np.array([p['probabilities'][1] for p in prediction])
