from __future__ import print_function
import glob, os, sys, cv2, random, time
import tensorflow as tf
import numpy as np
import imageio as io
from utils.utils_lap_pyramid import *
from utils.configs import *
from utils.utilities import *
#import matplotlib.pyplot as plt


'''Parameters to modify'''
level = '4'
'''===================='''


def gen_tfrec(lev):
    tfrecord_path = config.eval.tfrecord_eval + lev + '_' + config.model.tfrecord_suffix
    #tfrecord_path = config.eval.tfrecord_demo + lev + '_' + config.model.tfrecord_suffix
    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
        """####################### hdr ldr image reading ############################"""
        file_list = glob.glob(config.eval.hdr_path + '*.{}'.format('hdr'))
        #file_list = glob.glob(config.eval.demo_path + '*.{}'.format('hdr'))

        # finding corresponding ldr image
        length_images = len(file_list)
        for index in range(len(file_list)):
            start_img_time = time.time()
            cur_path = file_list[index]
            print( 'Processing Image -> ' + cur_path, ' %d / %d' % (index + 1, length_images))
            file_name = os.path.splitext(os.path.basename(cur_path))[0]

            # Read in the corresponding HDR Image
            hdr_img = io.imread(cur_path)

            # the image size is 3301 x 7768, too big to handle.  Resize it too 1/4
            hdr_img = cv2.resize(hdr_img, (config.data.width / 2, config.data.height / 2))

            # grayscale
            hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)

            # data preprocessing
            hdr_gray_clipped = cut_dark_end_percent(hdr_gray, 0.001)
            hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
            # log(np.finfo(float).eps) is -52.  so we clipped -50 off
            hdr_preprocessed = np.clip(hdr_logged, a_min=-50, a_max=np.max(hdr_logged))

            ''' bring to [0,1] '''
            hdr_ready = norm_0_to_1(hdr_preprocessed)

            '''############################ create laplacian pyramid ############################'''
            hdr_py = lpyr_gen(hdr_ready, int(lev))

            hdr_py_aligned, hdr_levs = lpyr_enlarge_to_top_but_bottom(hdr_py)

            hdr_py_aligned = dualize(hdr_py_aligned)

            '''############################ store in tfrecord ############################'''
            example = pack_example(hdr_py_aligned, file_name)
            tfrecord_writer.write(example.SerializeToString())

            elapsed_img = time.time() - start_img_time
            predict_finish_time = (length_images - index) * elapsed_img # in seconds
            the_time = complete_time_predict(predict_finish_time)
            print('\n')
            print('--------------------------------------------------------------------------------------------------')
            print('Processed Image -> ' + cur_path, ' %d / %d, took: %s' % (index+1, length_images, complete_time_predict(elapsed_img)))
            print('--------------------------------------------------------------------------------------------------')
            print('|||||Expected completion time: ' + the_time + '|||||')
            print('--------------------------------------------------------------------------------------------------')


def dualize(py_layers):
    freq_layer = 0
    bottom_layer = py_layers[-1]
    freq_layers = py_layers[:-1]
    for item in range(0, len(freq_layers)):
        freq_layer += freq_layers[item]

    dual_layers = [freq_layer, bottom_layer]
    return dual_layers


def complete_time_predict(dot):
    minutes = 60  # seconds
    hours = minutes * 60  # minutes
    days = hours * 24

    if dot < minutes:
        return '%.2f seconds' % dot
    elif minutes <= dot < hours:
        return '{} minutes'.format(dot//minutes)
    elif hours <= dot < days:
        return '{} hours, {} minutes'.format(dot//hours, int(dot % hours)/minutes)
    else:
        return '{} days, {} hours'.format(dot//days, int(dot % days)/hours)


def pack_example(img_patch, name):
    shape1 = img_patch[0].shape
    h1 = shape1[0]
    w1 = shape1[1]

    shape2 = img_patch[1].shape
    h2 = shape2[0]
    w2 = shape2[1]

    features = {}
    # store meta-data
    features['name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[name]))
    features['h1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h1]))
    features['w1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w1]))
    features['h2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h2]))
    features['w2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w2]))

    for l in range(0, len(img_patch)):
        img_patch[l] = np.reshape(img_patch[l], -1)
        features['eval{0}'.format(l)] = tf.train.Feature(float_list=tf.train.FloatList(value=img_patch[l]))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


gen_tfrec(level)


