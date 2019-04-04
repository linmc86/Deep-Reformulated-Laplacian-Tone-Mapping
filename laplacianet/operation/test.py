# coding: utf-8
from __future__ import print_function, division
import os, sys, cv2, glob
from utils.configs import *
from utils.utilities import *
from utils.utils_lap_pyramid import *
from data_parser.parse_tfrec import *
import net.net_new_structure as ns
import imageio as io
#import matplotlib.pyplot as plt


'''Parameters to modify'''
level = '4'
mode = 'demo' # 'demo' or 'test'
'''===================='''


pad_width = 10
paddings = tf.constant([[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])

hdr_dir = config.eval.hdr_path
model_ckp = config.model.ckp_path_high
tfrecord_path = config.eval.tfrecord_eval

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def setconf(lev, test):
    global model_ckp, tfrecord_path, hdr_dir

    if test == 'test':
        hdr_dir = config.eval.hdr_path
        tfrecord_path = config.eval.tfrecord_eval + lev + '_' + config.model.tfrecord_suffix
        model_ckp = config.model.ckp_path_ft
    elif test == 'demo':
        hdr_dir = config.eval.demo_path
        tfrecord_path = config.eval.tfrecord_demo + lev + '_' + config.model.tfrecord_suffix
        model_ckp = config.model.ckp_path_demo
    else:
        sys.exit("Wrong requesting name! It has to be either 'test' or 'demo' mode.")


def evalfrontlayer():
    global paddings

    """Read tfrecord"""
    train_iter = eval_iterator_dual_gray(tfrecord_path)
    high, bot_gray, h, w, name = train_iter.get_next()

    high = tf.expand_dims(high, axis=0)
    bot_gray = tf.expand_dims(bot_gray, axis=0)

    high = tf.expand_dims(high, axis=3)
    bot_gray = tf.expand_dims(bot_gray, axis=3)

    '''padding the bottom layer to refrain the ripple-boarder effect'''
    bot_gray = tf.pad(bot_gray, paddings, "REFLECT")

    return high, bot_gray, h, w, name


def restoreftlayer():
    high, bot, h, w, name = evalfrontlayer()

    ##################
    """Feed Network"""
    ##################
    out_h = ns.nethighlayer(high)
    out_b = ns.netbotlayer(bot)
    return out_h, out_b, high, bot, h, w, name


def main(lev, mode):
    global model_ckp, tfrecord_path, hdr_dir

    setconf(lev, mode)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        output_high, output_bot, input_high, input_bot, h, w, name = restoreftlayer()

        bot_shape = tf.shape(output_bot)
        output_bot = tf.slice(output_bot, [0, pad_width, pad_width, 0], [-1, bot_shape[1]-pad_width*2, bot_shape[2]-pad_width*2, -1])

        bot_h, bot_w = tfcalshape(h, w, lev)
        output_bot = tf.squeeze(output_bot)
        tfbot_upsampling = tf.reshape(output_bot, [bot_h, bot_w])

        i = tf.constant(0)
        n = tf.constant(int(lev))
        fullsize_bottom, i, n = tf.while_loop(cond, body, [tfbot_upsampling, i, n],
                                              shape_invariants=[tf.TensorShape([None, None]), i.get_shape(),
                                                                n.get_shape()])

        fullsize_bottom = tf.slice(fullsize_bottom, [0, 0], [h, w])
        fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=0)
        fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=3)

        imgpatch = output_high + fullsize_bottom

        output = ns.netftlayer(imgpatch)

        '''load network'''
        variables_to_restore = []
        for v in tf.global_variables():
            if not (v.name.startswith(config.model.loss_model)):
                variables_to_restore.append(v)

        saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
        ckpt = tf.train.get_checkpoint_state(model_ckp)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(model_ckp)
            saver.restore(sess, full_path)

        counter = 0
        num_eval_imgs = len(glob.glob(hdr_dir + '*.{}'.format('hdr')))

        while counter < num_eval_imgs:
            predict, filename = sess.run([output, name])
            predict = np.squeeze(predict)

            # Normalize to [0, 255]
            predict = norm_0_to_255(predict)

            hdr_path = hdr_dir + filename + '.hdr'
            hdr_img = io.imread(hdr_path)
            hdr_img = cv2.resize(hdr_img, (np.shape(predict)[1], np.shape(predict)[0]))
            hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)

            # bring back to RGB
            recovered_ldr = lum2rgb(predict, hdr_gray, hdr_img)
            recovered_ldr[recovered_ldr > 255] = 255
            recovered_ldr[recovered_ldr < 0] = 0

            # store results
            hdr_save_path = config.eval.result
            recovered_ldr = norm_0_to_255(recovered_ldr)

            io.imwrite(hdr_save_path + filename + '_predict.png', recovered_ldr)

            counter += 1


def tfcalshape(h, w, lev_scale):
    new_h, new_w = h, w
    for i in range(int(lev_scale)):
        new_h = tf.cast(tf.ceil(tf.divide(new_h, 2)), tf.int32)
        new_w = tf.cast(tf.ceil(tf.divide(new_w, 2)), tf.int32)
    return new_h, new_w


main(level, mode)









