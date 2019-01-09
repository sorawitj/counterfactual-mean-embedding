import os

import tensorflow as tf


class PolicyGradientAgent(object):
    def __init__(self, config, sess, initial_weight=None):
        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, config['context_dim'] + 1])

        initializer = None
        if initial_weight is not None:
            initializer = tf.constant_initializer(initial_weight)

        self.logits = tf.layers.dense(
            inputs=self._input,
            units=config['n_items'],
            kernel_initializer=initializer)

        self.action_probs = tf.nn.softmax(self.logits)

        self._action_dist = tf.distributions.Multinomial(total_count=1., probs=self.action_probs)
        self._max_action = tf.argmax(self.logits, axis=1)
        self._max_action_prob = tf.one_hot(self._max_action, depth=config['n_items'])

        # get log probabilities
        self.log_prob = tf.log(tf.nn.softmax(self.logits) + 1e-20)

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._rewards = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        self.indices = tf.range(0, tf.shape(self.log_prob)[0]) * tf.shape(self.log_prob)[1] + self._acts
        self.act_prob = tf.gather(tf.reshape(self.log_prob, [-1]), self.indices)

        # surrogate loss
        self.loss = -tf.reduce_mean(self.act_prob * self._rewards)

        # update + gradient clipping
        optimizer = tf.train.GradientDescentOptimizer(config['learning_rate'])
        self._train = optimizer.minimize(self.loss)

    def act(self, sample_users):
        # get one action, by sampling
        return self._s.run([self._action_dist.sample(1), self.action_probs],
                           feed_dict={self._input: sample_users})
        # return self._s.run([self._max_action, self._max_action_prob],
        #                    feed_dict={self._input: sample_users})

    def train_step(self, obs, acts, reward):
        batch_feed = {self._input: obs,
                      self._acts: acts,
                      self._rewards: reward}
        _, loss = self._s.run([self._train, self.loss], feed_dict=batch_feed)
        return loss


class PolicyGradientGaussian(object):
    def __init__(self, config, sess, initial_w=None):
        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, config['context_dim']])

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        w_initializer = None
        if initial_w is not None:
            w_initializer = tf.constant_initializer(initial_w)

        self.mu = tf.layers.dense(self._input,
                                  units=1,
                                  activation=None,
                                  use_bias=False,
                                  kernel_initializer=w_initializer,
                                  name='weight')
        self.mu = tf.squeeze(self.mu)

        with tf.variable_scope('weight', reuse=True):
            self.weights = tf.get_variable('kernel')

        self.action_dist = tf.distributions.Normal(self.mu, 1.0)

        # training part of graph
        self._acts = tf.placeholder(tf.float32)
        self._rewards = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        self.act_prob = tf.log(self.action_dist.prob(self._acts) + 1e-30)

        # surrogate loss
        self.loss = -tf.reduce_mean(self.act_prob * self._rewards)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self._train = optimizer.minimize(self.loss)

    def act(self, sample_users):
        # get one action, by sampling
        sample = self.action_dist.sample()
        action_prob = self.action_dist.prob(sample)
        return self._s.run([sample, action_prob],
                           feed_dict={self._input: sample_users})

    def train_step(self, obs, acts, reward, learning_rate):
        batch_feed = {self._input: obs,
                      self._acts: acts,
                      self._rewards: reward,
                      self.learning_rate: learning_rate}
        _, loss, weights = self._s.run([self._train, self.loss, self.weights], feed_dict=batch_feed)
        return loss, weights
