import cv2
import tensorflow as tf
import numpy as np


def gaussian_pyr(img,lev):
    img = img.astype(np.float32)
    g_pyr = [img]
    cur_g = img
    for index in range(lev):
        cur_g = cv2.pyrDown(cur_g)
        g_pyr.append(cur_g)
    return g_pyr


# generate the laplacian pyramid from an image with specified number of levels
def lpyr_gen(img,lev=2):
    img = img.astype(np.float32)  # this changes whatever the img dtype to 'float64'
    g_pyr = gaussian_pyr(img,lev)
    l_pyr = []
    for index in range(lev):
        cur_g = g_pyr[index]
        img_shape = np.shape(cur_g)
        cur_w = img_shape[0]
        cur_h = img_shape[1]
        next_g = cv2.pyrUp(g_pyr[index+1],dstsize=(cur_h,cur_w))
        cur_l = cv2.subtract(cur_g,next_g)
        l_pyr.append(cur_l)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def lpyr_recons(l_pyr):
    lev = len(l_pyr)
    cur_l = l_pyr[-1]
    for index in range(lev-2,-1,-1):
        img_shape = np.shape(l_pyr[index])
        next_w = img_shape[0]
        next_h = img_shape[1]
        cur_l = cv2.pyrUp(cur_l,dstsize=(next_h,next_w))
        next_l = l_pyr[index]
        cur_l = cur_l + next_l
    return cur_l


def lpyr_upsample(l_img,levels):
    cur_l = l_img
    lev = len(levels)-1
    for index in range(lev):
        h = levels[lev - 1 - index][0]
        w = levels[lev - 1 - index][1]
        cur_l = cv2.pyrUp(cur_l, dstsize=(w, h))
    return cur_l


# make all levels of pyramid to the same size as the largest one
def lpyr_enlarge_to_top(l_pyr):
    lev = len(l_pyr)
    levels = []
    cur_l=[]
    for index in range(lev):
        levels.append((l_pyr[index].shape))
        aligned = lpyr_upsample(l_pyr[index], levels)
        cur_l.append(aligned)
    return cur_l


# make all levels of pyramid to the same size as the largest one but the bottom layer
def lpyr_enlarge_to_top_but_bottom(l_pyr):
    lev = len(l_pyr)
    levels = []
    cur_l=[]
    for index in range(lev-1):
        levels.append((l_pyr[index].shape))
        aligned = lpyr_upsample(l_pyr[index], levels)
        cur_l.append(aligned)
    cur_l.append(l_pyr[lev-1])
    return cur_l, levels


# only upsamples one layer the bottom layer to specific size
def lpyr_enlarge_bottom_to_top(l_pyr, levels):
    levels.append((l_pyr[-1].shape))
    upsampled = lpyr_upsample(l_pyr[-1], levels)
    l_pyr[-1] = upsampled
    return l_pyr


def call2dtensorgaussfilter():
    return tf.constant([[1./256., 4./256., 6./256., 4./256., 1./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [6./256., 24./256., 36./256., 24./256., 6./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [1./256., 4./256., 6./256., 4./256., 1./256.]])


def applygaussian(imgs):
    gauss_f = call2dtensorgaussfilter()
    gauss_f = tf.expand_dims(gauss_f, axis=2)
    gauss_f = tf.expand_dims(gauss_f, axis=3)

    result = tf.nn.conv2d(imgs, gauss_f * 4, strides=[1, 1, 1, 1], padding="VALID")
    result = tf.squeeze(result, axis=0)
    result = tf.squeeze(result, axis=2)
    return result


def dilatezeros(imgs):
    zeros = tf.zeros_like(imgs)
    column_zeros = tf.reshape(tf.stack([imgs, zeros], 2), [-1, tf.shape(imgs)[1] + tf.shape(zeros)[1]])[:,:-1]

    row_zeros = tf.transpose(column_zeros)

    zeros = tf.zeros_like(row_zeros)
    dilated = tf.reshape(tf.stack([row_zeros, zeros], 2), [-1, tf.shape(row_zeros)[1] + tf.shape(zeros)[1]])[:,:-1]
    dilated = tf.transpose(dilated)

    paddings = tf.constant([[0, 1], [0, 1]])
    dilated = tf.pad(dilated, paddings, "REFLECT")

    dilated = tf.expand_dims(dilated, axis=0)
    dilated = tf.expand_dims(dilated, axis=3)
    return dilated


# funcs for tf.while_loop ====================================
def body(output_bot, i, n):
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    output_bot = dilatezeros(output_bot)
    output_bot = tf.pad(output_bot, paddings, "REFLECT")
    output_bot = applygaussian(output_bot)
    return output_bot, tf.add(i, 1), n


def cond(output_bot, i, n):
    return tf.less(i, n)

















