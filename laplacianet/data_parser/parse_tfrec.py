import tensorflow as tf
from utils.configs import *


def _parse_function_new_high(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train1': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'label1': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
        }
    )
    img = features['train1']
    label = features['label1']

    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size, 1])
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size, 1])

    return img, label


def _parse_function_new_bot(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'label2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img = features['train2']
    label = features['label2']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    img = tf.reshape(img, [h2, w2, 1])
    label = tf.reshape(label, [h2, w2, 1])

    return img, label


def _parse_function_new_ft(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train1': tf.FixedLenFeature(shape=(256 * 256,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(256 * 256,), dtype=tf.float32),
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img_h = features['train1']
    img_b = features['train2']
    label = features['label']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    img_h = tf.reshape(img_h, [256, 256, 1])
    img_b = tf.reshape(img_b, [h2, w2, 1])
    label = tf.reshape(label,  [256, 256, 1])

    return img_h, img_b, label


# eval, gray, dual
def _parse_eval_function(example_proto):
    """
    parse dual layer for evaluation.  h1, w1 are the size of high layer. h2, w2 are the size of bottom layer.
    :param example_proto:
    :return:
    """
    feature_labels = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h1': tf.FixedLenFeature([], dtype=tf.int64),
        'w1': tf.FixedLenFeature([], dtype=tf.int64),
        'h2': tf.FixedLenFeature([], dtype=tf.int64),
        'w2': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, 2):
        feature_labels['eval{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)
    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    eval0 = features['eval{0}'.format(0)]
    eval0 = tf.reshape(eval0, [h1, w1])

    eval1 = features['eval{0}'.format(1)]
    eval1 = tf.reshape(eval1, [h2, w2])

    return eval0, eval1, h1, w1, name


def data_iterator_new_high(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_high)
    data = data.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(config.train.batch_size_high).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_bot(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_bot)
    data = data.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(config.train.batch_size_bot).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_ft)
    data = data.shuffle(buffer_size=500, reshuffle_each_iteration=True).batch(config.train.batch_size_ft).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def eval_iterator_dual_gray(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater

