import tensorflow as tf


class PolicyGradientAgent(object):
    def __init__(self, config, sess, initial_weight=None):
        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, config['context_dim']])

        initializer = None
        if initial_weight is not None:
            initializer = tf.constant_initializer(initial_weight)

        self.logits = tf.layers.dense(
            inputs=self._input,
            units=config['n_items'],
            kernel_initializer=initializer)

        self.action_probs = tf.nn.softmax(self.logits)

        self._action_dist = tf.distributions.Multinomial(total_count=1., probs=self.action_probs)

        # get log probabilities
        self.log_prob = tf.log(tf.nn.softmax(self.logits) + 1e-10)

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._rewards = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        self.indices = tf.range(0, tf.shape(self.log_prob)[0]) * tf.shape(self.log_prob)[1] + self._acts
        self.act_prob = tf.gather(tf.reshape(self.log_prob, [-1]), self.indices)

        # surrogate loss
        self.loss = -tf.reduce_mean(self.act_prob * self._rewards)

        # update + gradient clipping
        optimizer = tf.train.AdamOptimizer(config['learning_rate'])
        self._train = optimizer.minimize(self.loss)

    def act(self, sample_users):
        # get one action, by sampling
        return self._s.run([self._action_dist.sample(1), self.action_probs],
                           feed_dict={self._input: sample_users})

    def train_step(self, obs, acts, reward):
        batch_feed = {self._input: obs,
                      self._acts: acts,
                      self._rewards: reward}
        _, loss = self._s.run([self._train, self.loss], feed_dict=batch_feed)
        return loss
