from loss.custom_vgg16 import *
import tensorflow as tf
from utils.configs import *
#import matplotlib.pyplot as plt


class cal_loss(object):

    def __init__(self, img, gt, vgg_path, sess, withtv=False):
        self.data_dict = loadWeightsData(vgg_path)

        """Build Perceptual Losses"""
        with tf.name_scope(name=config.model.loss_model + "_run_vgg16"):
            # content target feature
            vgg_c = custom_Vgg16(gt, data_dict=self.data_dict)
            fe_generated = [vgg_c.conv1_1, vgg_c.conv2_1, vgg_c.conv3_1, vgg_c.conv4_1, vgg_c.conv5_1]

            # feature after transformation
            vgg = custom_Vgg16(img, data_dict=self.data_dict)
            fe_input = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]

        with tf.name_scope(name=config.model.loss_model + "_cal_content_L"):
            # compute feature loss
            loss_f = 0
            for f_g, f_i in zip(fe_generated, fe_input):
                loss_f += tf.reduce_mean(tf.abs(f_g - f_i))
        self.loss_f = loss_f

        """Build Total Variation Loss"""
        self.loss_tv = 0
        if withtv:
            shape = tf.shape(img)
            height = shape[1]
            width = shape[2]
            y = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(img, [0, 1, 0, 0], [-1, -1, -1, -1])
            x = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(img, [0, 0, 1, 0], [-1, -1, -1, -1])
            # self.loss_tv = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
            self.loss_tv = tf.reduce_sum(tf.abs(x)) / tf.to_float(tf.size(x)) + tf.reduce_sum(tf.abs(y)) / tf.to_float(tf.size(y))

        # total loss
        self.loss = self.loss_f + self.loss_tv



