
from __future__ import print_function
import functools
import json
import logging
import os
import sys
import nltk
import csv
import errno

import numpy as np
import tensorflow as tf


DATADIR = './data'


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


# Logging
tf.logging.set_verbosity(logging.INFO)
mkdir('results')
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_before, line_after):
    # Encode in Bytes for TF
    source = [c for c in line_before.rstrip('\n')]
    t = [w for w in nltk.word_tokenize(line_after.strip())]
    learn_target = t + ['<eos>'] + ['<pad>']
    target = ['<go>'] + t + ['<eos>']
    return (source, len(source)), (target, learn_target, len(target))


def generator_fn(data_file):
    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            yield parse_fn(row['before'], row['after'])


def input_fn(data_file, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), ([None], [None], ()))
    types = ((tf.string, tf.int32), (tf.string, tf.string, tf.int32))
    defaults = (('<pad>', 0), ('<pad>', '<pad>', 0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file), output_shapes=shapes, output_types=types)
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 5), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['source'], features['source_length']

    # Read vocabs and inputs
    dropout = params['dropout']
    source, source_length = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_source = tf.contrib.lookup.index_table_from_file(vocabulary_file=params['source_vocab_file'], num_oov_buckets=params['num_oov_buckets'])
    #num_sources = vocab_source.size()
    with open(params['source_vocab_file']) as f:
        num_sources = sum(1 for _ in f) + params['num_oov_buckets']
    vocab_target = tf.contrib.lookup.index_table_from_file(vocabulary_file=params['target_vocab_file'], num_oov_buckets=params['num_oov_buckets'])
    #num_targets = vocab_target.size()
    with open(params['target_vocab_file']) as f:
        num_targets = sum(1 for _ in f) + params['num_oov_buckets']
    target, learn_target, target_length = labels

    # source embeddings matrix
    _source_embedding = tf.Variable(tf.random_uniform([num_sources, params['embedding_size']]))
    source_ids = vocab_source.lookup(source)
    source_embedding = tf.nn.embedding_lookup(_source_embedding, source_ids)

    # target embeddings matrix
    target_embedding = tf.Variable(tf.random_uniform([num_targets, params['embedding_size']]))
    target_ids = vocab_target.lookup(target)
    target_learn_ids = vocab_target.lookup(learn_target)

    # --- encoder ---

    # add multilayer bidirectional RNN
    cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, source_embedding, sequence_length=source_length, dtype=tf.float32)

    # prepare output
    output = tf.concat(outputs, axis=-1)
    encoder_output = tf.layers.dense(output, params['dim'])

    # prepare state
    state_fw, state_bw = states
    cells = []
    for fw, bw in zip(state_fw, state_bw):
        state = tf.concat([fw, bw], axis=-1)
        cells += [tf.layers.dense(state, params['dim'])]
    encoder_state = tuple(cells)

    # remember batch size
    batch_size = tf.shape(source_ids)[0]

    # --- decoder ---

    # add multilayer RNN cell
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])

    # add attention layer
    attention = tf.contrib.seq2seq.BahdanauAttention(num_units=params['dim'], memory=encoder_output, memory_sequence_length=source_length)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, attention_layer_size=params['dim'])
    decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    # projection layer
    projection_layer = tf.layers.Dense(num_targets, use_bias=False)

    # prediction encoder
    prediction_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=target_embedding,
        start_tokens=tf.fill([batch_size], tf.to_int32(vocab_target.lookup(tf.fill([], '<go>')))),
        end_token=tf.to_int32(vocab_target.lookup(tf.fill([], '<eos>'))))
    prediction_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, prediction_helper, decoder_initial_state, output_layer=projection_layer)
    prediction_output, _, _ = tf.contrib.seq2seq.dynamic_decode(prediction_decoder, maximum_iterations=params['max_iters'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        reverse_vocab_target = tf.contrib.lookup.index_to_string_table_from_file(params['target_vocab_file'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(prediction_output.sample_id))
        predictions = {
            'ids': output.sample_id,
            'text': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # train encoder
        _target_embedding = tf.nn.embedding_lookup(target_embedding, target_ids)
        train_helper = tf.contrib.seq2seq.TrainingHelper(_target_embedding, target_length)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, decoder_initial_state, output_layer=projection_layer)
        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
        train_output = train_outputs.rnn_output
        train_sample_id = train_outputs.sample_id

        # loss
        masks = tf.sequence_mask(lengths=target_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits=train_output, targets=target_learn_ids, weights=masks)

        metrics = {
            'acc': tf.metrics.accuracy(target_ids, train_sample_id, masks),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 128,
        'embedding_size': 100,
        'max_iters': 50,
        'dropout': 0.5,
        'layers': 3,
        'num_oov_buckets': 3,
        'epochs': 15,
        'batch_size': 50,
        'buffer': 15000,
        'source_vocab_file': os.path.join(DATADIR, 'vocab.source.txt'),
        'target_vocab_file': os.path.join(DATADIR, 'vocab.target.txt'),
    }
    with open('results/params.json', 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'train.csv'), params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'dev.csv'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    mkdir(estimator.eval_dir())
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'acc', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        mkdir('results/score')
        with open('results/score/{}.preds.txt'.format(name), 'wb') as f:
            path = os.path.join(DATADIR, '{}.csv'.format(name))
            test_inpf = functools.partial(input_fn, path)
            golds_gen = generator_fn(path)
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((source, _), (target, _, _)) = golds
                f.write('{}\t {}\t{}\n'.format(''.join(source), ' '.join(target), ' '.join(preds['text'])))

    for name in ['test', 'dev']:
        write_predictions(name)

