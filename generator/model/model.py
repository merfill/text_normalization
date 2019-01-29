import numpy as np
import os
import tensorflow as tf
from tensorflow.python.util import nest

from .data_utils import *
from .general_utils import Progbar
from .base_model import BaseModel


class GenModel(BaseModel):
    def __init__(self, config):
        super(GenModel, self).__init__(config)
        self.go_char_index = self.config.char_vocab_size
        self.eos_char_index = self.config.char_vocab_size + 1
        self.config.words_vocab[0] = '<pad>'


    def add_placeholders(self):
        # shape = (batch size, max length of input clause in chars)
        self.source_ids = tf.placeholder(tf.int32, shape=[None, None], name="source_ids")

        # shape = (batch size, max length of output clause in words)
        self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name="target_ids")
        self.target_learn_ids = tf.placeholder(tf.int32, shape=[None, None], name="target_learn_ids")

        # shape = (batch_size)
        self.source_lengths = tf.placeholder(tf.int32, shape=[None], name="source_lengths")
        self.target_lengths = tf.placeholder(tf.int32, shape=[None], name="target_lengths")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # dynamic calculated batch size
        self.batch_size = tf.shape(self.source_ids)[0]


    def get_feed_dict(self, before_clauses, after_clauses=None, lr=None, dropout=None):
        # perform padding of the given data
        source_ids, source_lengths = pad_sequences(sequences=before_clauses, pad_tok=PAD,
                                                                go=self.go_char_index, eos=self.eos_char_index)

        # build feed dictionary
        feed = {
            self.source_ids: source_ids,
            self.source_lengths: source_lengths,
        }

        if after_clauses is not None:
            feed[self.target_ids], feed[self.target_lengths] =\
                pad_sequences(sequences=after_clauses, pad_tok=PAD, go=self.config.vocab_words[GO], eos=self.config.vocab_words[EOS])
            feed[self.target_learn_ids], _ =\
                pad_sequences(sequences=after_clauses, pad_tok=PAD, eos=self.config.vocab_words[EOS])

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_encoder_op(self):
        with tf.variable_scope('encoder'):
            # get char embeddings matrix
            _embedding = tf.Variable(tf.random_uniform([self.config.char_vocab_size+2, self.config.char_embedding_size]))
            embedding = tf.nn.embedding_lookup(_embedding, self.source_ids)

            # add multilayer bidirectional RNN
            cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.config.num_hidden) for _ in range(self.config.num_encoder_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.config.num_hidden) for _ in range(self.config.num_encoder_layers)])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding, sequence_length=self.source_lengths, dtype=tf.float32)

            # prepare output
            output = tf.concat(outputs, axis=-1)
            self.encoder_output = tf.layers.dense(output, self.config.num_hidden)

            # prepare state
            state_fw, state_bw = states
            cells = []
            for fw, bw in zip(state_fw, state_bw):
                state = tf.concat([fw, bw], axis=-1)
                cells += [tf.layers.dense(state, self.config.num_hidden)]
            self.encoder_state = tuple(cells)


    def add_decoder_cell_op(self):
        with tf.variable_scope("decoder_cell"):
            # get word embedding matrix
            self.target_embedding = tf.Variable(tf.random_uniform([self.config.word_vocab_size, self.config.word_embedding_size]))

            # add multilayer RNN cell
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.config.num_hidden) for _ in range(self.config.num_encoder_layers)])

            encoder_output = self.encoder_output
            encoder_state = self.encoder_state
            encoder_length = self.source_lengths
            batch_size = self.batch_size

            if self.config.use_beamsearch:
                encoder_output = tf.contrib.seq2seq.tile_batch(self.encoder_output, multiplier=self.config.beam_width)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.config.beam_width), self.encoder_state)
                encoder_length = tf.contrib.seq2seq.tile_batch(self.source_lengths, multiplier=self.config.beam_width)
                batch_size = batch_size * self.config.beam_width

            # add attention layer if any
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.config.num_hidden, memory=encoder_output,
                                                                            memory_sequence_length=encoder_length)
            self.decoder_initial_state = encoder_state
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.config.num_hidden)
            self.decoder_initial_state = self.decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            # Add projection layer
            self.projection_layer = tf.layers.Dense(self.config.word_vocab_size, use_bias=False)


    def add_train_decoder_op(self):
        with tf.variable_scope("train_decoder"):
            embeddings = tf.nn.embedding_lookup(self.target_embedding, self.target_ids)
            helper = tf.contrib.seq2seq.TrainingHelper(embeddings, self.target_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state, output_layer=self.projection_layer)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            self.train_output = output.rnn_output
            self.train_sample_id = output.sample_id


    def add_greedy_decoder_op(self):
        with tf.variable_scope("greedy_decoder"):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.target_embedding,
                start_tokens=tf.fill([self.batch_size], self.config.vocab_words[GO]),
                end_token=self.config.vocab_words[EOS])
            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state, output_layer=self.projection_layer)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.config.max_iterations)
            self.greedy_output = output.sample_id


    def add_beamsearch_decoder_op(self):
        with tf.variable_scope("beamsearch_decoder"):
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.target_embedding,
                start_tokens=tf.fill([self.batch_size], self.config.vocab_words[GO]),
                end_token=self.config.vocab_words[EOS],
                initial_state=self.decoder_initial_state,
                beam_width=self.config.beam_width,
                output_layer=self.projection_layer)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.config.max_iterations)
            self.beamsearch_output = output.predicted_ids[0]


    def add_loss_op(self):
        with tf.variable_scope('loss'):
            masks = tf.sequence_mask(lengths=self.target_lengths, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_output, targets=self.target_learn_ids, weights=masks)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        self.add_placeholders()
        self.add_encoder_op()
        self.add_decoder_cell_op()
        if self.config.use_beamsearch:
            self.add_beamsearch_decoder_op()
        else:
            self.add_train_decoder_op()
            self.add_greedy_decoder_op()
            self.add_loss_op()
            self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)

        # Generic functions that add training op and initialize session
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, before_clauses):
        fd = self.get_feed_dict(before_clauses, dropout=1.)
        if self.config.use_beamsearch:
            return self.sess.run(self.beamsearch_output, feed_dict=fd)
        else:
            return self.sess.run(self.greedy_output, feed_dict=fd)


    def run_epoch(self, train, dev, epoch, print_test=False):
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (before_clauses, after_clauses) in enumerate(minibatches(train, batch_size)):
            fd = self.get_feed_dict(before_clauses, after_clauses, self.config.lr, self.config.dropout)
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        self.logger.info(' - acc: {:04.2f}'.format(metrics['acc']))
        if print_test:
            for err in metrics['errors']:
                self.logger.info('{} => {} ? {}'.format(err[0], err[1], err[2]))

        return metrics["acc"]


    def to_text(self, ids, chars=True):
        if chars:
            delim = ''
            vocab = self.config.chars_vocab
            go = self.go_char_index
            eos = self.eos_char_index
        else:
            delim = ' '
            vocab = self.config.words_vocab
            go = self.config.vocab_words[GO]
            eos = self.config.vocab_words[EOS]
        r = []
        for idx in ids:
            if idx == eos:
                break
            elif idx == go:
                continue
            r += [vocab[idx]]
        return delim.join(r)


    def run_evaluate(self, test):
        accs = []
        errors = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for before, after in minibatches(test, self.config.batch_size):
            pred  = self.predict_batch(before)
            for before, after, pred in zip(before, after, pred):
                was_error = False
                for (a, b) in zip(pred, after):
                    if a == b:
                        accs += [1]
                    else:
                        accs += [0]
                        was_error = True
                if was_error:
                    errors += [(self.to_text(before), self.to_text(after, chars=False), self.to_text(pred, chars=False))]
        acc = np.mean(accs)

        return {"acc": 100*acc, 'errors': errors}


    def predict(self, test):
        fd = self.get_feed_dict(test)
        pred = self.predict_batch(before)
        return [self.to_text(pred, chars=False)]
