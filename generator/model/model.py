import numpy as np
import os
import tensorflow as tf


from .data_utils import *
from .general_utils import Progbar
from .base_model import BaseModel


class GenModel(BaseModel):
    def __init__(self, config):
        super(GenModel, self).__init__(config)
        self.go_char_index = self.config.char_vocab_size
        self.eos_char_index = self.config.char_vocab_size + 1


    def add_placeholders(self):
        # shape = (batch size, max length of input clause in chars)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None], name="char_ids")

        # shape = (batch size, max length of output clause in words)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        # shape = (batch_size)
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="input_sequence_lengths")
        self.output_sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="output_sequence_lengths")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # dynamic calculated batch size
        self.batch_size = tf.shape(self.word_ids)[0]


    def get_feed_dict(self, before_clauses, after_clauses, lr=None, dropout=None):
        # perform padding of the given data
        char_ids, char_sequence_lengths = pad_sequences(sequences=before_clauses, pad_tok=PAD,
                                                            go=self.go_char_index, eos=self.eos_char_index)
        word_ids, word_sequence_lengths = pad_sequences(sequences=after_clauses, pad_tok=PAD, go=self.config.vocab_words[GO],
                                                            eos=self.config.vocab_words[EOS])

        # build feed dictionary
        feed = {
            self.char_ids: char_ids,
            self.input_sequence_lengths: char_sequence_lengths,
            self.word_ids: word_ids,
            self.output_sequence_lengths: word_sequence_lengths,
        }

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_encoder_op(self):
        with tf.variable_scope('encoder'):
            # get char embeddings matrix
            char_embeddings = tf.Variable(tf.random_uniform([self.config.char_vocab_size+2, self.config.char_embedding_size]))
            self.char_embeddings = tf.nn.embedding_lookup(char_embeddings, self.char_ids)

            # add multilayer RNN
            cells = [tf.contrib.rnn.GRUCell(self.config.num_hidden) for _ in range(self.config.num_encoder_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(cell, self.char_embeddings,
                                                                            sequence_length=self.input_sequence_lengths, dtype=tf.float32)


    def add_decoder_cell_op(self):
        with tf.variable_scope("decoder_cell"):
            # get word embedding matrix
            self.word_embedding_decoder = tf.Variable(tf.random_uniform([self.config.word_vocab_size, self.config.word_embedding_size]))
            self.word_embeddings = tf.nn.embedding_lookup(self.word_embedding_decoder, self.word_ids)

            # add multilayer RNN
            cells = [tf.contrib.rnn.GRUCell(self.config.num_hidden) for _ in range(self.config.num_encoder_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            # add attention layer if any
            if False and self.config.use_attention:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.num_hidden, self.encoder_output, memory_sequence_length=None)
                cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.config.num_hidden)
                self.decoder_initial_state = cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)
            else:
                self.decoder_initial_state = self.encoder_state

            self.decoder_cell = cell

            # Add projection layer
            self.projection_layer = tf.layers.Dense(self.config.word_vocab_size, use_bias=False)


    def add_train_decoder_op(self):
        with tf.variable_scope("train_decoder"):
            helper = tf.contrib.seq2seq.TrainingHelper(self.word_embeddings, self.output_sequence_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state, output_layer=self.projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            self.decoder_train_output = outputs.rnn_output


    def add_inference_decoder_op(self):
        with tf.variable_scope("inference_decoder"):
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(self.decoder_initial_state, multiplier=self.config.beam_width)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.word_embedding_decoder,
                start_tokens=tf.fill([self.batch_size], self.config.vocab_words[GO]),
                end_token=self.config.vocab_words[EOS],
                initial_state=decoder_initial_state,
                beam_width=self.config.beam_width,
                output_layer=self.projection_layer,
                length_penalty_weight=.0)
            maximum_iterations = tf.round(tf.reduce_max(self.output_sequence_lengths) * 2)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
            self.decoder_predicted_ids = outputs.predicted_ids


    def add_loss_op(self):
        with tf.variable_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_train_output, labels=self.word_ids)
            mask = tf.sequence_mask(self.output_sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        self.add_placeholders()
        self.add_encoder_op()
        self.add_decoder_cell_op()
        self.add_train_decoder_op()
        self.add_inference_decoder_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, before_clauses, after_clauses):
        fd = self.get_feed_dict(before_clauses, after_clauses, dropout=1.0)
        words_pred = self.sess.run(self.decoder_predicted_ids, feed_dict=fd)
        return words_pred


    def run_epoch(self, train, dev, epoch):
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
        for err in metrics['errors']:
            self.logger.info('{} => {} ? {}'.format(err[0], err[1], err[2]))

        return metrics["acc"]


    def to_text(self, ids, chars=True):
        if chars:
            delim = ''
            vocab = seld.char_vocab
        else:
            delim = ' '
            vocab = self.word_vocab
        r = []
        for idx in ids:
            r += [vocab[idx]]
        return delim.join(r)


    def run_evaluate(self, test):
        accs = []
        errors = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for before, after in minibatches(test, self.config.batch_size):
            pred  = self.predict_batch(before, after)

            print pred
            print '---------'
            print before
            print '=========='
            print after
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


#    def predict(self, clauses_raw):
        #clauses = [self.config.processing_clause(w) for w in clauses_raw]
        #if type(clauses[0]) == tuple:
        #    clauses = zip(*clauses)
        #pred_ids, _ = self.predict_batch([clauses])
        #preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        #return preds