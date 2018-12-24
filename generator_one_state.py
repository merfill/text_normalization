
import tensorflow as tf
import numpy as np
import random
import math
import sys
import collections
import nltk
import json
import codecs
from pprint import pprint


class TextGenerator:

    def __init__(self, char_embedding_size, char_embeddings, char_to_id, start_char_id, end_char_id,
                  word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word, start_word_id,
                  end_word_id, in_seq_len=100, out_seq_len=50, state_size=128, num_layers=3,
                  learning_rate=.003, max_grad_norm=.5, state_stack_value=3):
        # General parameters
        self._state_size = state_size
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._max_grad_norm = max_grad_norm
        self._state_stack_value = state_stack_value

        # Character specific parameters
        self._in_seq_len = in_seq_len
        self._char_embedding_size = char_embedding_size
        self._char_embeddings = char_embeddings
        self._char_to_id = char_to_id
        self._start_char_id = start_char_id
        self._end_char_id = end_char_id

        # Word specific parameters
        self._out_seq_len = out_seq_len
        self._word_embedding_size = word_embedding_size
        self._word_embeddings = word_embeddings
        self._word_one_hots = word_one_hots
        self._word_to_id = word_to_id
        self._id_to_word = id_to_word
        self._start_word_id = start_word_id
        self._end_word_id = end_word_id
        self._num_words = self._word_one_hots.shape[1]

        # Placeholders
        self._char_input = tf.placeholder(tf.float32, [None, self._in_seq_len, self._char_embedding_size])
        self._word_input = tf.placeholder(tf.float32, [None, self._out_seq_len, self._word_embedding_size])
        self._target = tf.placeholder(tf.float32, [None, self._out_seq_len, self._num_words])

        # Graph definition
        self._init_in_length()
        self._init_out_length()
        self._init_prediction()
        self._init_cost()
        self._init_error()
        self._init_optimize()


    @classmethod
    def create_from_data(cls, char_data, word_data, char_embedding_size=32, word_embedding_size=32,
                          in_seq_len=100, out_seq_len=50, state_size=128, num_layers=3,
                          learning_rate=.003, max_grad_norm=.5, state_stack_value=3):
        # Initialize character data
        char_to_id = {}
        for text in char_data:
            for c in text:
                if c not in char_to_id:
                    id = len(char_to_id)
                    char_to_id[c] = id
        vocab_size = len(char_to_id) + 2
        char_embeddings = np.random.random([vocab_size, char_embedding_size])
        start_char_id = vocab_size - 2
        end_char_id = vocab_size - 1

        # Initialize word data
        word_to_id = {}
        id_to_word = {}
        for text in word_data:
            for word in nltk.word_tokenize(text):
                if word not in word_to_id:
                    id = len(word_to_id)
                    word_to_id[word] = id
                    id_to_word[id] = word
        vocab_size = len(word_to_id) + 2 # Add two extra 'pseudo words': start and end
        word_one_hots = np.eye(vocab_size)
        word_embeddings = np.random.random([vocab_size, word_embedding_size])
        start_word_id = vocab_size - 2
        end_word_id = vocab_size - 1

        return cls(char_embedding_size, char_embeddings, char_to_id, start_char_id, end_char_id, 
                   word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word,
                   start_word_id, end_word_id, in_seq_len, out_seq_len, state_size, num_layers,
                   learning_rate, max_grad_norm, state_stack_value)


    @classmethod
    def load_params(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as f:
            data = json.load(f)

        # General parameters
        state_size = data['state_size']
        num_layers = data['num_layers']
        learning_rate = data['learning_rate']
        max_grad_norm = data['max_grad_norm']
        state_stack_value = data['state_stack_value']

        # Character specific parameters
        in_seq_len = data['in_seq_len']
        char_embedding_size = data['char_embedding_size']
        char_embeddings = np.array(data['char_embeddings'])
        char_to_id = data['char_to_id']
        start_char_id = data['start_char_id']
        end_char_id = data['end_char_id']

        # Word specific parameters
        out_seq_len = data['out_seq_len']
        word_embedding_size = data['word_embedding_size']
        word_embeddings = np.array(data['word_embeddings'])
        word_one_hots = np.array(data['word_one_hots'])
        word_to_id = data['word_to_id']

        id_to_word = {}
        for word_id in data['id_to_word']:
            id_to_word[int(word_id)] = data['id_to_word'][word_id]

        start_word_id = data['start_word_id']
        end_word_id = data['end_word_id']

        return cls(char_embedding_size, char_embeddings, char_to_id, start_char_id, end_char_id, 
                   word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word,
                   start_word_id, end_word_id, in_seq_len, out_seq_len, state_size, num_layers,
                   learning_rate, max_grad_norm, state_stack_value)


    def save_params(self, file_path):
        data = {}

        # General parameters
        data['state_size'] = self._state_size
        data['num_layers'] = self._num_layers
        data['learning_rate'] = self._learning_rate
        data['max_grad_norm'] = self._max_grad_norm
        data['state_stack_value'] = self._state_stack_value

        # Character specific parameters
        data['in_seq_len'] = self._in_seq_len
        data['char_embedding_size'] = self._char_embedding_size
        data['char_embeddings'] = self._char_embeddings.tolist()
        data['char_to_id'] = self._char_to_id
        data['start_char_id'] = self._start_char_id
        data['end_char_id'] = self._end_char_id

        # Word specific parameters
        data['out_seq_len'] = self._out_seq_len
        data['word_embedding_size'] = self._word_embedding_size
        data['word_embeddings'] = self._word_embeddings.tolist()
        data['word_one_hots'] = self._word_one_hots.tolist()
        data['word_to_id'] = self._word_to_id
        data['id_to_word'] = self._id_to_word
        data['start_word_id'] = self._start_word_id
        data['end_word_id'] = self._end_word_id

        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


    def _init_in_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self._char_input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        self._in_length = tf.cast(length, tf.int32)


    def _init_out_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self._word_input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        self._out_length = tf.cast(length, tf.int32)


    def _init_prediction(self):
        # Batch size
        batch_size = tf.shape(self._char_input)[0]

        # Input recurrent network, we don't use state
        with tf.variable_scope('input_rnn'):
            in_cells = []
            for _ in range(self._num_layers):
                in_cells.append(tf.contrib.rnn.GRUCell(self._state_size))
            in_cell = tf.contrib.rnn.MultiRNNCell(in_cells)

            in_output, _ = tf.nn.dynamic_rnn(in_cell, self._char_input,
                                                dtype=tf.float32, sequence_length=self._in_length)

        # State layer
        with tf.variable_scope('state'):
            # Get relevant output
            index = tf.range(0, batch_size) * self._in_seq_len + (self._in_length - 1)
            flat = tf.reshape(in_output, [-1, self._state_size])
            self._state = tf.gather(flat, index)

        tf.summary.histogram("state", self._state)

        # Output recurrent network
        with tf.variable_scope('output_rnn'):
            out_cells = []
            state_list = []
            for _ in range(self._num_layers):
                out_cells.append(tf.contrib.rnn.GRUCell(self._state_size))
                state_list.append(self._state)
            out_cell = tf.contrib.rnn.MultiRNNCell(out_cells)

            state_type = type(out_cell.zero_state(batch_size, tf.float32))
            self._initial_state = state_type([
                tf.placeholder_with_default(state, [None, self._state_size]) for state in state_list])

            out_output, self._final_state = tf.nn.dynamic_rnn(out_cell, self._word_input,
                                              dtype=tf.float32, sequence_length=self._out_length,
                                              initial_state=self._initial_state)

        # Softmax output
        with tf.variable_scope('generation'):
            out_weight = tf.get_variable('out_W', [self._state_size, self._num_words])
            out_bias = tf.get_variable('out_b', [self._num_words], initializer=tf.constant_initializer(0.1))
            out_output = tf.reshape(out_output, [-1, self._state_size])
            self._raw_logits = tf.matmul(out_output, out_weight) + out_bias
            self._logits = tf.nn.softmax(self._raw_logits)
            self._prediction = tf.reshape(self._logits, [-1, self._out_seq_len, self._num_words])


    def _init_cost(self):
        with tf.variable_scope('cost'):
            # Mask to drop paddings
            mask = tf.sign(tf.reduce_max(tf.abs(self._target), reduction_indices=2))

            # Compute cross entropy for prediction
            cross_entropy = self._target * tf.log(self._prediction + 1e-20)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            cross_entropy *= mask

            # Average over actual sequence lengths
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(self._out_length, tf.float32)

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
            mistakes /= tf.cast(self._out_length, tf.float32)
            self._error = tf.reduce_mean(mistakes)
            tf.summary.scalar('prediction_error', self._error)


    def _init_optimize(self):
        with tf.variable_scope('optimize'):
            tvars = tf.trainable_variables()
            grads = tf.gradients(self._cost, tvars)
            clip_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
            self._grad_norm = tf.global_norm(grads)
            tf.summary.scalar('gradients', self._grad_norm)
            optimizer = tf.train.AdamOptimizer(self._learning_rate)

            self._optimize = optimizer.apply_gradients(zip(clip_grads, tvars))


    def _gen_char_seq(self, text, gen_start=True, gen_end=True):
        if len(text) > self._in_seq_len - 2:
            text = text[:self._in_seq_len - 2]

        seq = [self._char_embeddings[self._char_to_id[c]] for c in text]
        if gen_start:
            seq = [self._char_embeddings[self._start_char_id]] + seq
        if gen_end:
            seq = seq + [self._char_embeddings[self._end_char_id]]
        seq = seq + ([[0] * self._char_embedding_size] * (self._in_seq_len - len(seq)))

        return seq


    def _gen_word_seq(self, text, embeddings, gen_start=True, gen_end=True):
        words = nltk.word_tokenize(text)
        if len(words) > (self._out_seq_len - 2):
            raise Exception('to long output text: ', word_text)
        elif len(words) == 0:
            raise Exception('zero output text: ', word_text)

        seq = [embeddings[self._word_to_id[word]] for word in words]
        if gen_start:
            seq = [embeddings[self._start_word_id]] + seq
        if gen_end:
            seq = seq + [embeddings[self._end_word_id]]
        seq = seq + ([[0] * embeddings.shape[1]] * (self._out_seq_len - len(seq)))

        return seq


    def _get_batch(self, data, indices):
        char_input = [self._gen_char_seq(text) for text in data[:,0][indices]]
        word_input = [self._gen_word_seq(text, self._word_embeddings) for text in data[:,1][indices]]
        word_output = [self._gen_word_seq(text, self._word_one_hots, gen_start=False) for text in data[:,1][indices]]

        return (np.array(char_input), np.array(word_input), np.array(word_output))


    def sample(self, sess, char_input):
        state = sess.run(self._initial_state, feed_dict={self._char_input: char_input})

        word_input_seq = np.array([self._word_embeddings[self._start_word_id]]
                            + [[0] * self._word_embedding_size for _ in range(self._out_seq_len - 1)])
        word_input = np.array([word_input_seq.copy() for _ in range(char_input.shape[0])])

        state, prediction = sess.run([self._final_state, self._prediction],
                                     feed_dict={self._word_input: word_input, self._initial_state: state})

        batch_size = prediction.shape[0]
        word_output = np.zeros([batch_size, self._out_seq_len, self._word_embedding_size])

        ends = [False] * prediction.shape[0]
        texts = ['' for _ in range(prediction.shape[0])]
        for i in range(self._out_seq_len - 1):
            preds = np.argmax(prediction, 2)
            for j in range(preds.shape[0]):
                if ends[j]:
                    next

                word_id = preds[j][0]
                word_output[j][i] = np.copy(self._word_embeddings[word_id])
                if word_id == self._end_word_id:
                    ends[j] = True
                    word_input[j] = np.copy(word_input_seq)
                else:
                    if len(texts[j]) > 0:
                        texts[j] += ' '
                    texts[j] += self._id_to_word[word_id]
                    word_input[j][0] = np.copy(self._word_embeddings[word_id])

            state, prediction = sess.run([self._final_state, self._prediction],
                                     feed_dict={self._word_input: word_input, self._initial_state: state})

        return (np.array(texts), word_output)


    def train(self, log_dir, model_dir, num_steps, train, validation, batch_size, restore_weights=False):
        sess = tf.Session()

        with tf.variable_scope('error'):
            gen_error = tf.get_variable('generation_error', [], initializer=tf.constant_initializer(1.))

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if restore_weights:
            saver.restore(sess, model_dir + '/best.chkp')
            print 'Model parameters were restored from {0} directory'.format(model_dir)

        tf.summary.scalar('gen_error', gen_error)
        summaries = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

        train_writer.add_graph(sess.graph)

        error_buffer = collections.deque(maxlen=50)
        max_error = 1e20
        sample_step = 0
        for step in range(num_steps):
            train_indicies = np.random.choice(len(train), size=batch_size, replace=False)

            char_input, word_input, target = self._get_batch(train, train_indicies)
            _, s = sess.run([self._optimize, summaries],
                    feed_dict={self._char_input: char_input, self._word_input: word_input, self._target: target})
            train_writer.add_summary(s, step)
            train_writer.flush()

            validation_indicies = np.random.choice(len(validation), size=batch_size, replace=False)
            char_data = validation[validation_indicies][:,0]
            word_data = validation[validation_indicies][:,1]
            char_input, word_input, target = self._get_batch(validation, validation_indicies)

            texts, _ = self.sample(sess, char_input)

            print '\n{0} --> {1} <==> {2}\n'.format(char_data[0].encode('utf-8'), texts[0].strip().encode('utf-8'),
                                                    word_data[0].strip().encode('utf-8'))

            equals = 0.
            for i in range(len(texts)):
                  if texts[i].strip() == word_data[i].strip():
                    equals += 1
            ge = 1. - equals / len(texts)

            sess.run(gen_error.assign(ge))
            s, e = sess.run([summaries, self._error],
                    feed_dict={self._char_input: char_input, self._word_input: word_input, self._target: target})

            test_writer.add_summary(s, step)
            test_writer.flush()

            sys.stdout.write('\r{0} {1} {2}'.format(step, e, ge))

            # Save best mean error value from last 50 iterations
            error_buffer.append(e)
            mean_error = np.mean(np.array(error_buffer))
            if max_error > mean_error:
                saver.save(sess, model_dir + '/best.chkp')
                max_error = mean_error
                print 'save best model for: ', max_error


    def generate(self, sess, data):
        char_input = np.array([self._gen_char_seq(text) for text in data])
        state = sess.run(self._initial_state, feed_dict={self._char_input: char_input})

        word_input_seq = np.array([self._word_embeddings[self._start_word_id]]
                            + [[0] * self._word_embedding_size for _ in range(self._out_seq_len - 1)])
        word_input = np.array([word_input_seq.copy() for _ in range(char_input.shape[0])])

        state, prediction = sess.run([self._final_state, self._prediction],
                                     feed_dict={self._word_input: word_input, self._initial_state: state})

        batch_size = prediction.shape[0]

        ends = [False] * prediction.shape[0]
        texts = ['' for _ in range(prediction.shape[0])]
        for i in range(self._out_seq_len - 1):
            preds = np.argmax(prediction, 2)
            for j in range(preds.shape[0]):
                if ends[j]:
                    next

                word_id = preds[j][0]
                if word_id == self._end_word_id:
                    ends[j] = True
                    word_input[j] = np.copy(word_input_seq)
                else:
                    if len(texts[j]) > 0:
                        texts[j] += ' '
                    texts[j] += self._id_to_word[word_id]
                    word_input[j][0] = np.copy(self._word_embeddings[word_id])

            state, prediction = sess.run([self._final_state, self._prediction],
                                     feed_dict={self._word_input: word_input, self._initial_state: state})

        return np.array(texts)

