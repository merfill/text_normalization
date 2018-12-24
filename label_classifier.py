
import tensorflow as tf
import numpy as np
import random
import math
import sys
import collections
import json
import codecs
from pprint import pprint

class LabelClassifier:

    def __init__(self, embedding_size, embeddings, char_to_id, start_char_id, end_char_id, label_one_hot, num_classes,
                 id_to_label, label_to_id, state_size=128, seq_len=100, num_layers=3, learning_rate=.003, max_grad_norm=.5):
        self._state_size = state_size
        self._seq_len = seq_len
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._max_grad_norm = max_grad_norm
        self._label_one_hot = label_one_hot
        self._num_classes = num_classes
        self._id_to_label = id_to_label
        self._label_to_id = label_to_id

        # Character specific parameters
        self._embedding_size = embedding_size
        self._embeddings = embeddings
        self._char_to_id = char_to_id
        self._start_char_id = start_char_id
        self._end_char_id = end_char_id

        self._input = tf.placeholder(tf.float32, [None, self._seq_len, self._embedding_size], name='input')
        self._target = tf.placeholder(tf.float32, [None, self._num_classes], name='target')

        self._init_length()
        self._init_prediction()
        self._init_cost()
        self._init_error()
        self._init_optimize()
        self._init_classify()


    @classmethod
    def create_from_data(cls, data, label_list, embedding_size=32, state_size=128, seq_len=100, num_layers=3,
                            learning_rate=.003, max_grad_norm=.5):
        # Initialize labels
        label_one_hot = {}
        id_to_label = {}
        label_to_id = {}
        label_one_hot = np.eye(len(label_list)).tolist()
        for i in range(len(label_list)):
            label_to_id[label_list[i]] = i
            id_to_label[i] = label_list[i]

        # Initialize character data
        char_to_id = {}
        for text in data:
            for c in text:
                if c not in char_to_id:
                    char_id = len(char_to_id)
                    char_to_id[c] = char_id
        vocab_size = len(char_to_id) + 2
        embeddings = np.random.random([vocab_size, embedding_size]).tolist()
        start_char_id = vocab_size - 2
        end_char_id = vocab_size - 1

        return cls(embedding_size, embeddings, char_to_id, start_char_id, end_char_id, label_one_hot, len(label_one_hot),
                    id_to_label, label_to_id, state_size, seq_len, num_layers, learning_rate, max_grad_norm)


    @classmethod
    def load_params(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as f:
            data = json.load(f)

            state_size = data['state_size']
            seq_len = data['seq_len']
            num_layers = data['num_layers']
            learning_rate = data['learning_rate']
            max_grad_norm = data['max_grad_norm']
            label_one_hot = data['label_one_hot']
            num_classes = data['num_classes']

            id_to_label = {}
            for char_id in data['id_to_label']:
                id_to_label[int(char_id)] = data['id_to_label'][char_id]
            label_to_id = data['label_to_id']

            # Character specific parameters
            embedding_size = data['embedding_size']
            embeddings = data['embeddings']
            char_to_id = data['char_to_id']
            start_char_id = data['start_char_id']
            end_char_id = data['end_char_id']

        return cls(embedding_size, embeddings, char_to_id, start_char_id, end_char_id, label_one_hot, num_classes, id_to_label,
                    label_to_id, state_size, seq_len, num_layers, learning_rate,  max_grad_norm)


    def save_params(self, file_path):
        data = {}

        data['state_size'] = self._state_size
        data['seq_len'] = self._seq_len
        data['num_layers'] = self._num_layers
        data['learning_rate'] = self._learning_rate
        data['max_grad_norm'] = self._max_grad_norm
        data['label_one_hot'] = self._label_one_hot
        data['num_classes'] = self._num_classes
        data['id_to_label'] = self._id_to_label
        data['label_to_id'] = self._label_to_id

        # Character specific parameters
        data['embedding_size'] = self._embedding_size
        data['embeddings'] = self._embeddings
        data['char_to_id'] = self._char_to_id
        data['start_char_id'] = self._start_char_id
        data['end_char_id'] = self._end_char_id

        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


    def _init_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self._input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        self._length = tf.cast(length, tf.int32)


    def _init_prediction(self):
        # Dimensions
        self._max_length = int(self._input.get_shape()[1])
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

        # Get relevant output
        index = tf.range(0, batch_size) * self._max_length + (self._length - 1)
        flat = tf.reshape(self._output, [-1, self._state_size])
        relevant = tf.gather(flat, index)

        # Prediction layer
        with tf.variable_scope('prediction'):
            weight = tf.get_variable('W', [self._state_size, self._num_classes])
            bias = tf.get_variable('b', [self._num_classes], initializer=tf.constant_initializer(0.1))
            self._logits = tf.matmul(relevant, weight) + bias
            self._prediction = tf.nn.softmax(self._logits)


    def _init_cost(self):
        with tf.variable_scope('cost'):
            self._cost = tf.losses.softmax_cross_entropy(self._target, self._logits)
        tf.summary.scalar('cost', self._cost)


    def _init_error(self):
        with tf.variable_scope('error'):
            mistakes = tf.not_equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, dtype=tf.float32))

        tf.summary.scalar('prediction_error', self._error)

    def _init_classify(self):
        with tf.variable_scope('classify'):
            self._classify = tf.argmax(self._prediction, 1)


    def _init_optimize(self):
        tvars = tf.trainable_variables()
        grads = tf.gradients(self._cost, tvars)
        clip_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._learning_rate)

        self._optimize = optimizer.apply_gradients(zip(clip_grads, tvars))


    def _get_seq(self, text):
        if len(text) > self._seq_len - 2:
            text = text[:self._seq_len - 2]

        seq = [self._embeddings[self._char_to_id[c]] for c in text]
        seq = [self._embeddings[self._start_char_id]] + seq
        seq = seq + [self._embeddings[self._end_char_id]]
        seq = seq + ([[0] * self._embedding_size] * (self._seq_len - len(seq)))

        return seq


    def _get_batch(self, data, batch_size):
        indicies = np.random.choice(len(data), size=batch_size, replace=False)

        y = [self._label_one_hot[self._label_to_id[label]] for label in data[:,1][indicies]]
        x = [self._get_seq(text) for text in data[:,0][indicies]]

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


    def classify(self, model_dir, data, batch_size):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, model_dir + '/best.chkp')

        result = []
        num_batches = len(data) // batch_size
        print 'number of iterations: ', num_batches
        for batch_index in range(num_batches):
            print 'processing batch: ', batch_index
            batch_data = data[batch_index * batch_size: batch_index * batch_size + batch_size]
            char_input = np.array([self._get_seq(text) for text in batch_data])

            label_ids = sess.run(self._classify, feed_dict={self._input: char_input})
            for label_id in label_ids:
                result.append(self._id_to_label[label_id])

        return result


    def generate(self, sess, data):
        char_input = np.array([self._get_seq(text) for text in data])
        predictions = sess.run(self._prediction, feed_dict={self._input: char_input})
        labels = np.argsort(predictions, axis=1)
        return [[self._id_to_label[label_id] for label_id in label_ids[::-1]] for label_ids in labels[:,-3:]]


