
import tensorflow as tf
import numpy as np
import math
import json
import codecs
from pprint import pprint
import collections
import sys

class LabelLangModel:

    def __init__(self, label_embedding_size, label_embeddings, start_sentence_id, end_sentence_id, label_one_hot,
                    id_to_label, label_to_id, state_size=128, seq_len=50, num_layers=3, learning_rate=.003, max_grad_norm=.5):
        self._label_embedding_size = label_embedding_size
        self._label_embeddings = label_embeddings
        self._start_sentence_id = start_sentence_id
        self._end_sentence_id = end_sentence_id
        self._label_one_hot = label_one_hot
        self._id_to_label = id_to_label
        self._label_to_id = label_to_id
        self._num_labels = len(label_one_hot)

        self._state_size = state_size
        self._seq_len = seq_len
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._max_grad_norm = max_grad_norm

        self._input = tf.placeholder(tf.float32, [None, self._seq_len, self._label_embedding_size], name='input')
        self._target = tf.placeholder(tf.float32, [None, self._seq_len, self._num_labels], name='target')

        self._init_length()
        self._init_prediction()
        self._init_cost()
        self._init_error()
        self._init_optimize()


    @classmethod
    def create_from_data(cls, label_list, label_embedding_size=32, state_size=128, seq_len=50, num_layers=3, learning_rate=.003,
                            max_grad_norm=.5):
        # Initialize labels
        label_one_hot = {}
        id_to_label = {}
        label_to_id = {}
        vocab_size = len(label_list) + 2
        start_sentence_id = vocab_size - 2
        end_sentence_id = vocab_size - 1
        for i in range(len(label_list)):
            label_to_id[label_list[i]] = i
            id_to_label[i] = label_list[i]

        label_one_hot = np.eye(vocab_size).tolist()
        label_embeddings = np.random.random([vocab_size, label_embedding_size]).tolist()

        return cls(label_embedding_size, label_embeddings, start_sentence_id, end_sentence_id, label_one_hot, id_to_label,
                    label_to_id, state_size, seq_len, num_layers, learning_rate, max_grad_norm)


    @classmethod
    def restore_from_file(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as f:    
            data = json.load(f)

            label_embedding_size = data['label_embedding_size']
            label_embeddings = data['label_embeddings']
            start_sentence_id = data['start_sentence_id']
            end_sentence_id = data['end_sentence_id']
            label_one_hot = data['label_one_hot']
            id_to_label = {}
            for char_id in data['id_to_label']:
                id_to_label[int(char_id)] = data['id_to_label'][char_id]
            label_to_id = data['label_to_id']

            state_size = data['state_size']
            seq_len = data['seq_len']
            num_layers = data['num_layers']
            learning_rate = data['learning_rate']
            max_grad_norm = data['max_grad_norm']

        return cls(label_embedding_size, label_embeddings, start_sentence_id, end_sentence_id, label_one_hot, id_to_label,
                    label_to_id, state_size, seq_len, num_layers, learning_rate, max_grad_norm)


    def save(self, file_path):
        data = {}

        data['label_embedding_size'] = self._label_embedding_size
        data['label_embeddings'] = self._label_embeddings
        data['start_sentence_id'] = self._start_sentence_id
        data['end_sentence_id'] = self._end_sentence_id
        data['label_one_hot'] = self._label_one_hot
        data['id_to_label'] = self._id_to_label
        data['label_to_id'] = self._label_to_id
        data['state_size'] = self._state_size
        data['seq_len'] = self._seq_len
        data['num_layers'] = self._num_layers
        data['learning_rate'] = self._learning_rate
        data['max_grad_norm'] = self._max_grad_norm

        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


    def _init_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self._input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        self._length = tf.cast(length, tf.int32)


    def _init_prediction(self):
        # Dimensions
        batch_size = tf.shape(self._input)[0]

        # Recurrent network
        with tf.variable_scope('rnn'):
            cells = []
            for _ in range(self._num_layers):
                cells.append(tf.contrib.rnn.GRUCell(self._state_size))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            states = cell.zero_state(batch_size, tf.float32)
            state_type = type(states)
            self._initial_state = [
                tf.placeholder_with_default(zero_state, [None, self._state_size]) for zero_state in states]
            self._initial_state = state_type(self._initial_state)

            self._output, self._final_state = tf.nn.dynamic_rnn(cell, self._input,
                                                                dtype=tf.float32, sequence_length=self._length,
                                                                initial_state=self._initial_state)

        # Prediction layer
        with tf.variable_scope('prediction'):
            # Softmax layer.
            weight = tf.get_variable('W', [self._state_size, self._num_labels])
            bias = tf.get_variable('b', [self._num_labels], initializer=tf.constant_initializer(0.1))

            # Flatten to apply same weights to all time steps.
            output = tf.reshape(self._output, [-1, self._state_size])
            self._raw_logits = tf.matmul(output, weight) + bias
            self._logits = tf.nn.softmax(self._raw_logits)
            self._prediction = tf.reshape(self._logits, [-1, self._seq_len, self._num_labels])


    def _init_cost(self):
        with tf.variable_scope('cost'):
            # Compute cross entropy for each frame.
            cross_entropy = self._target * tf.log(self._prediction + 1e-20)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            mask = tf.sign(tf.reduce_max(tf.abs(self._target), reduction_indices=2))
            cross_entropy *= mask

            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(self._length, tf.float32)

            self._cost = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cost', self._cost)


    def _init_error(self):
        with tf.variable_scope('error'):
            mistakes = tf.not_equal(tf.argmax(self._target, 2), tf.argmax(self._prediction, 2))
            mistakes = tf.cast(mistakes, tf.float32)
            mask = tf.sign(tf.reduce_max(tf.abs(self._target), reduction_indices=2))
            mistakes *= mask

            # Average over actual sequence lengths.
            mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
            mistakes /= (tf.cast(self._length, tf.float32) + 1e-20)
            self._error = tf.reduce_mean(mistakes)
            tf.summary.scalar('prediction_error', self._error)


    def _init_optimize(self):
        tvars = tf.trainable_variables()
        grads = tf.gradients(self._cost, tvars)
        clip_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._learning_rate)

        self._optimize = optimizer.apply_gradients(zip(clip_grads, tvars))


    def _get_in_seq(self, label_list, insert_start_id=True, insert_end_id=True):
        if len(label_list) > self._seq_len - 2:
            label_list = label_list[:self._seq_len - 2]

        seq = [self._label_embeddings[self._label_to_id[label]] for label in label_list]
        if insert_start_id:
            seq = [self._label_embeddings[self._start_sentence_id]] + seq
        if insert_end_id:
            seq = seq + [self._label_embeddings[self._end_sentence_id]]
        seq = seq + ([[0] * self._label_embedding_size] * (self._seq_len - len(seq)))

        return seq


    def _get_out_seq(self, label_list, insert_start_id=True, insert_end_id=True):
        if len(label_list) > self._seq_len - 2:
            label_list = label_list[:self._seq_len - 2]

        seq = [self._label_one_hot[self._label_to_id[label]] for label in label_list]
        if insert_start_id:
            seq = [self._label_one_hot[self._start_sentence_id]] + seq
        if insert_end_id:
            seq = seq + [self._label_one_hot[self._end_sentence_id]]
        seq = seq + ([[0] * self._num_labels] * (self._seq_len - len(seq)))

        return seq

    def _get_batch(self, data, batch_size):
        indicies = np.random.choice(len(data), size=batch_size, replace=False)

        x = [self._get_in_seq(label_list, True) for label_list in data[indicies]]
        y = [self._get_out_seq(label_list, False) for label_list in data[indicies]]

        return (np.array(x), np.array(y))


    def train(self, log_dir, model_dir, num_steps, train, validation, batch_size):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
        summaries = tf.summary.merge_all()

        train_writer.add_graph(sess.graph)
        saver = tf.train.Saver()

        error_buffer = collections.deque(maxlen=50)
        max_error = 1e20
        for step in range(num_steps):
            x, y = self._get_batch(train, batch_size)
            _, s = sess.run([self._optimize, summaries], feed_dict={self._input: x, self._target: y})
            train_writer.add_summary(s, step)
            train_writer.flush()

            x, y = self._get_batch(validation, batch_size)
            s, e = sess.run([summaries, self._error], feed_dict={self._input: x, self._target: y})
            test_writer.add_summary(s, step)
            test_writer.flush()
            sys.stdout.write('\r{0} {1}'.format(step, e))

            # Save best mean error value from last 50 iterations
            error_buffer.append(e)
            mean_error = np.mean(np.array(error_buffer))
            if max_error > mean_error:
                saver.save(sess, model_dir + '/best.chkp')
                max_error = mean_error
                print 'save best model for: ', max_error


    def sample_next(self, sess, state, labels):
        if state is None:
            state = sess.run(self._initial_state, feed_dict={self._char_input: char_input})

        x = np.array([self._get_in_seq(label, False, False) for label in labels])
        state, prediction = sess.run([self._final_state, self._prediction],
                                        feed_dict={self._input: x, self._initial_state: state})
        result_labels = [self.id_to_label[label_id] for label_id in np.argmax(prediction, axis=1)]
        return (result_labels, state)

