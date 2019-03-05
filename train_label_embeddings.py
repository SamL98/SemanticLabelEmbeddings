import tensorflow as tf
from functools import partial
from argparse import ArgumentParser
import os
from util import *

arg_parser = ArgumentParser(description='Train the label embeddings.')
arg_parser.add_argument('--data_dir', '-ddir', dest='data_dir', type=str, default=data_dir, help='The directory that contains the tfrecords files.')
arg_parser.add_argument('--batch_size', '-bsize', dest='batch_size', type=int, default=8, help='The batch size to use when training the embeddings.')
arg_parser.add_argument('--num_steps', '-nstep', dest='num_steps', type=int, default=10000, help='The number of steps to train the embeddings for.')
arg_parser.add_argument('--embedding_size', '-esize', dest='embedding_size', type=int, default=100, help='The dimensionality of the embeddings to train.')
args = arg_parser.parse_args()


def model_fn(features, labels, params):
    '''
    The model function for the embeddings estimator

    Params:
        features: A dictionary mapping feature names to tensors.
                    Should be just { 'given_label': label_tensor } where label_tensor has shape (1,) and dtype tf.uint8
        labels: A tensor of the context labels. Should just be a tf.uint8
        params: A dictionary of other parameters.
                    Only thing of note should be feature_columns which should be set to [input_key]

    Returns:
        An EstimatorSpec object for the embedding estimator
    '''

    embedding_size = params['embedding_size']
    nc = params['n_classes']

    inputs = tf.features_column.input_layer(features, params['feature_columns'])

    embeddings = tf.Variable(tf.random_uniform([nc, embedding_size], -1.0, 1.0))
    label_vectors = tf.nn.embedding_lookup(embeddings, inputs)

    context_matrix = tf.Variable(tf.random_uniform([embedding_size, nc], -1.0, 1.0))
    context_logits = tf.matmul(context_matrix, label_vectors)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, context_logits)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

def parse_example(tf_example):
    '''
    Parse the features and label for a given tensorflow example

    Params:
        tf_example: The serialized tensorflow protobuf example

    Returns:
        A tuple of the input features and corresponding label
    '''
    feats_dict = {
        'given_label': tf.FixedLenFeature((1), tf.int64, default_value=0),
        'context_label': tf.FixedLenFeature((1), tf.int64, default_value=0)
    }
    features = tf.parse_single_example(tf_example, feats_dict)
    return { given_tf_key, features[given_tf_key]}, features[context_tf_key]

def dataset_input_fn(fnames, batch_size):
    '''
    Input function for the estimator.

    Params:
        fnames: A list of the tfrecords filenames.
        batch_size: The batch size to use while training.

    Returns:
        A tensorflow Dataset object to use as the input function for the embeddings estimator.
    '''
    dataset = tf.data.TFRecordDataset(fnames)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


if __name__ == '__main__':
    fnames = [join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    input_fn = partial(dataset_input_fn, fnames, args.batch_size)

    feature_columns = [tf.feature_column.numeric_column(given_tf_key, dtype=tf.int64)]
    params = {
        'features_columns': feature_columns,
        'n_classes': nc ,
        'embedding_size': args.embedding_size
    }

    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=embeddings_model_dir)
    estimator.train(input_fn, steps=args.num_steps)