from __future__ import print_function
import glob, os, sys, cv2, random, time, math
import tensorflow as tf
import numpy as np
import imageio as io
from utils.utils_lap_pyramid import *
from utils.configs import *
from utils.utilities import *
import matplotlib.pyplot as plt


'''Parameters to modify'''
level = '4'
ft = False
'''===================='''



def gen_tfrec(lev, fine_tuning=False):
    tfrecord_path = config.model.tfrecord_dual + lev + '_' + config.model.tfrecord_suffix

    if fine_tuning:
        tfrecord_path = config.model.tfrecord_ft + lev + '_' + config.model.tfrecord_suffix

    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
        """####################### image reading ############################"""
        file_list = glob.glob(config.data.hdr_path + '*.{}'.format('hdr'))

        # finding corresponding ldr image
        length_images = len(file_list)
        for index in range(len(file_list)):
            start_img_time = time.time()
            cur_path = file_list[index]
            print('Processing Image -> ' + cur_path, ' %d / %d' % (index + 1, length_images))

            file_name = os.path.splitext(os.path.basename(cur_path))

            ldr_path = config.data.ldr_path + file_name[0] + '.jpg'

            # Read HDR LDR images
            hdr_img = io.imread(cur_path)
            ldr_img = io.imread(ldr_path)

            # the image size is 3301 x 7768, too big to handle.  Resize it too 1/4
            hdr_img = cv2.resize(hdr_img, (config.data.width/2, config.data.height/2))
            ldr_img = cv2.resize(ldr_img, (config.data.width/2, config.data.height/2))

            ''' check shape '''
            assert(np.shape(hdr_img) == np.shape(ldr_img))

            ''' grayscale '''
            hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
            ldr_gray = cv2.cvtColor(ldr_img, cv2.COLOR_RGB2GRAY)

            ''' data preprocessing '''
            hdr_gray_clipped = cut_dark_end_percent(hdr_gray, 0.001)
            hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
            # log(np.finfo(float).eps) is -52.  so we clipped -50 off
            hdr_preprocessed = np.clip(hdr_logged, a_min=-50, a_max=np.max(hdr_logged))

            ''' bring to [0,1] '''
            hdr_ready = norm_0_to_1(hdr_preprocessed)
            ldr_ready = norm_0_to_1(ldr_gray)

            '''############################ cropping images ############################'''
            if fine_tuning:
                img_rand_patches, label_rand_patches = crop_random(hdr_ready, ldr_ready,
                                                                   config.data.random_patch_ratio_x,
                                                                   config.data.random_patch_ratio_y,
                                                                   config.data.patch_size_ft,
                                                                   config.data.random_patch_per_img)
            else:
                img_rand_patches, label_rand_patches = crop_random(hdr_ready, ldr_ready,
                                                                   config.data.random_patch_ratio_x,
                                                                   config.data.random_patch_ratio_y,
                                                                   config.data.patch_size,
                                                                   config.data.random_patch_per_img)

            ''' check length '''
            assert (len(img_rand_patches) == len(label_rand_patches))

            '''############################ create laplacian pyramid ############################'''
            if fine_tuning:
                ''' Modify the pyramid - '''
                for i in range(len(img_rand_patches)):
                    img_rand_patches[i] = lpyr_gen(img_rand_patches[i], int(lev))

                ''' Modify the pyramid - '''
                for i in range(len(img_rand_patches)):
                    img_rand_patches[i], _ = lpyr_enlarge_to_top_but_bottom(img_rand_patches[i])

                ''' Add all high frequency layer up, making it only 2 layers, the high frequency layer and bottom layer. '''
                for i in range(len(img_rand_patches)):
                    img_rand_patches[i] = dualize(img_rand_patches[i])
            else:
                for i in range(len(img_rand_patches)):
                    img_rand_patches[i] = lpyr_gen(img_rand_patches[i], int(lev))
                    label_rand_patches[i] = lpyr_gen(label_rand_patches[i], int(lev))

                for i in range(len(img_rand_patches)):
                    img_rand_patches[i], _ = lpyr_enlarge_to_top_but_bottom(img_rand_patches[i])
                    label_rand_patches[i], _ = lpyr_enlarge_to_top_but_bottom(label_rand_patches[i])

                for i in range(len(img_rand_patches)):
                    img_rand_patches[i] = dualize(img_rand_patches[i])
                    label_rand_patches[i] = dualize(label_rand_patches[i])

                ''' check shape '''
                for i in range(len(img_rand_patches)):
                    assert (np.shape(img_rand_patches[i]) == np.shape(label_rand_patches[i]))

            ''' check length'''
            assert (len(img_rand_patches) == len(label_rand_patches))

            '''############################ store in tfrecord ############################'''
            patch_length = len(img_rand_patches)
            for i in range(0, patch_length):
                print('\r-- processing images patches %d / %d' % (i + 1, patch_length), end='')
                sys.stdout.flush()
                if fine_tuning:
                    example = pack_example(img_rand_patches[i], label_rand_patches[i], fine_tuning)
                else:
                    example = pack_example(img_rand_patches[i], label_rand_patches[i], fine_tuning)
                tfrecord_writer.write(example.SerializeToString())

            elapsed_img = time.time() - start_img_time
            predict_finish_time = (length_images - index) * elapsed_img  # in seconds
            the_time = complete_time_predict(predict_finish_time)
            print('\n')
            print('-------------------------------------------------------------------------------------------------')
            print('Processed Image -> ' + cur_path,
                  ' %d / %d, took: %s' % (index + 1, length_images, complete_time_predict(elapsed_img)))
            print('-------------------------------------------------------------------------------------------------')
            print('|||||Expected completion time: ' + the_time + '|||||')
            print('-------------------------------------------------------------------------------------------------')


def pack_example(img_patch, label_patch, fine_tuning=False):
    features = {}

    if fine_tuning:
        label_patch = np.reshape(label_patch, -1)
        features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_patch))
    else:
        label_high_patch = np.reshape(label_patch[0], -1)
        features['label1'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_high_patch))
        label_bot_patch = np.reshape(label_patch[1], -1)
        features['label2'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_bot_patch))

    h2,w2 = img_patch[1].shape

    features['h2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h2]))
    features['w2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w2]))

    ''' high layer patches size are fixed to be 512x512 '''
    img_high_patch = np.reshape(img_patch[0], -1)
    features['train1'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_high_patch))

    img_bot_patch = np.reshape(img_patch[1], -1)
    features['train2'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_bot_patch))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def complete_time_predict(dot):
    minutes = 60  # seconds
    hours = minutes * 60  # minutes
    days = hours * 24

    if dot < minutes:
        return '%.2f seconds' % dot
    elif minutes <= dot < hours:
        return '{} minutes'.format(dot // minutes)
    elif hours <= dot < days:
        return '{} hours, {} minutes'.format(dot // hours, int(dot % hours) / minutes)
    else:
        return '{} days, {} hours'.format(dot // days, int(dot % days) / hours)


def crop_random(img, label, x, y, size, N):
    imgpatchs = []
    labelpatchs = []
    h, w = np.shape(img)

    for i in range(N):
        rand_coe_h = random.random() * (y - x) + x
        rand_coe_w = random.random() * (y - x) + x

        # get width and height of the patch
        rand_h = int(h * rand_coe_h)
        rand_w = int(w * rand_coe_w)

        # the random - generated coordinates are limited in
        # h -> [0, coor_h]
        # w -> [0, coor_w]
        coor_h = h - rand_h
        coor_w = w - rand_w

        # get x and y starting point of the patch
        coor_x = int(random.random() * coor_h)
        coor_y = int(random.random() * coor_w)

        # only create patches for the high layer
        img_patch = img[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_img = cv2.resize(img_patch, (size, size))
        imgpatchs.append(resize_img)

        # Create patches for the label
        label_patch = label[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_label = cv2.resize(label_patch, (size, size))
        labelpatchs.append(resize_label)

    return imgpatchs, labelpatchs


def dualize(py_layers):
    freq_layer = 0
    bottom_layer = py_layers[-1]
    freq_layers = py_layers[:-1]
    for item in range(0, len(freq_layers)):
        freq_layer += freq_layers[item]

    dual_layers = [freq_layer, bottom_layer]
    return dual_layers


gen_tfrec(level, fine_tuning=ft)