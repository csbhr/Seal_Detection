"""

"""

import tensorflow as tf
import utils

FLAGS = utils.FLAGS
label_len = utils.label_len
num_class = utils.num_class


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        self.labels = tf.placeholder(tf.float32, [None, label_len, num_class])
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [FLAGS.image_channel, 64, 128, 128, FLAGS.out_channels]
        strides = [1, 2]

        feature_h = FLAGS.image_height
        feature_w = FLAGS.image_width

        count_ = 0
        min_size = min(FLAGS.image_height, FLAGS.image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count_ += 1

        # CNN part
        with tf.variable_scope('cnn'):
            x = self.inputs
            for i in range(FLAGS.cnn_count):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[1])
                    x = self._batch_norm('bn%d' % (i + 1), x)
                    x = self._leaky_relu(x, FLAGS.leakiness)
                    # x = self._max_pool(x, 2, strides[1])

                    _, feature_h, feature_w, _ = x.get_shape().as_list()

        # change feature_w to label_len
        x = tf.transpose(x, [0, 1, 3, 2])  # [batch_size, feature_h, FLAGS.out_channels, feature_w]
        x = tf.reshape(x, [-1, feature_w])  # [batch_size*feature_h*FLAGS.out_channels, feature_w]
        W_cl = tf.get_variable(name='W_change_label_len',
                               shape=[feature_w, label_len],
                               dtype=tf.float32,
                               initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
        b_cl = tf.get_variable(name='b_change_label_len',
                               shape=[label_len],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer())
        x = tf.matmul(x, W_cl) + b_cl  # [batch_size*feature_h*FLAGS.out_channels, label_len]
        x = tf.reshape(x, [FLAGS.batch_size, feature_h, FLAGS.out_channels,
                           label_len])  # [batch_size, feature_h, FLAGS.out_channels, label_len]
        x = tf.transpose(x, [0, 1, 3, 2])  # [batch_size, feature_h, label_len, FLAGS.out_channels]
        _, feature_h, feature_w, _ = x.get_shape().as_list()
        print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))

        # LSTM part
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in lstm.
            x = tf.reshape(x, [FLAGS.batch_size, feature_w, feature_h * FLAGS.out_channels])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            # print('self.seq_len.shape: {}'.format(self.seq_len.shape.as_list()))

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)

            cell1 = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(FLAGS.batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, FLAGS.num_hidden]

        # softmax part
        with tf.variable_scope('softmax'):
            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]

            W = tf.get_variable(name='W_out',
                                shape=[FLAGS.num_hidden, num_class],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[num_class],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            if self.mode == 'train':
                self.logits = tf.nn.dropout(self.logits, keep_prob=FLAGS.output_keep_prob)

            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_class])

            self.prob = tf.reshape(self.logits, [-1, num_class])
            self.prob = tf.nn.softmax(self.prob)
            self.prob = tf.reshape(self.prob, tf.shape(self.logits))

    def _build_train_op(self):
        # self.global_step = tf.Variable(0, trainable=False)
        self.global_step = tf.train.get_or_create_global_step()

        label_reshaped = tf.reshape(self.labels, [-1, num_class])
        logit_reshaped = tf.reshape(self.logits, [-1, num_class])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_reshaped,
                                                                              logits=logit_reshaped))
        tf.summary.scalar('loss', self.loss)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        prob_reshaped = tf.reshape(self.prob, [-1, num_class])
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(prob_reshaped, axis=1), tf.argmax(label_reshaped, axis=1)), dtype=tf.float32)
        )
        tf.summary.scalar('accuracy', self.accuracy)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = \
                tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=0.9,
                    center=True,
                    scale=True,
                    epsilon=1e-5,
                    updates_collections=None,
                    is_training=self.mode == 'train',
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=True,
                    scope='BatchNorm'
                )

        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
