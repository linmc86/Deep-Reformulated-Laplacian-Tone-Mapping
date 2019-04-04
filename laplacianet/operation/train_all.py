# coding: utf-8
from __future__ import print_function
from __future__ import division
from data_parser.parse_tfrec import *
from loss.cal_loss import *
from utils.utilities import *
from utils.utils_lap_pyramid import *
from utils.configs import *
import net.net_new_structure as ns
import time, os, sys
#import matplotlib.pyplot as plt

'''Parameters to modify'''
level = '4'
epochs = 50
'''===================='''



batchnum = config.train.batchnum_ft
model_ckp = config.model.ckp_path_ft
tfrecord_path = config.model.tfrecord_ft
height = 256
width = 256

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setconf(layer, lev):
    global model_ckp, tfrecord_path
    if layer == 'high':
        model_ckp = config.model.ckp_path_high
    elif layer == 'bot':
        model_ckp = config.model.ckp_path_bot
    elif layer == 'ft':
        model_ckp = config.model.ckp_path_ft
        tfrecord_path = config.model.tfrecord_ft + lev + '_' + config.model.tfrecord_suffix
    else:
        sys.exit("Wrong requesting layer name!")


def evalfrontlayer():
    """Read tfrecord"""
    train_iter = data_iterator_new_ft(tfrecord_path)
    high, bot, gt = train_iter.get_next()

    return high, bot, gt


def restoreftlayer():
    high, bot, gt = evalfrontlayer()

    """Feed Network"""
    out_h = ns.nethighlayer(high)
    out_b = ns.netbotlayer(bot)
    return out_h, out_b, high, bot, gt


def main(lev, goal_epoch):
    setconf('ft', lev)
    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

                output_high, output_bot, input_high, input_bot, gt = restoreftlayer()

                h, w = calshape(height, width, lev)
                tfbot_upsampling = tf.reshape(output_bot, [config.train.batch_size_ft, h, w])

                new_bot = 0
                for index in range(config.train.batch_size_ft):
                    fullsize_bottom = tf.squeeze(tf.slice(tfbot_upsampling, [index, 0, 0], [1, -1, -1]))

                    i = tf.constant(0)
                    n = tf.constant(int(lev))
                    fullsize_bottom, i, n = tf.while_loop(cond, body, [fullsize_bottom, i, n],
                                                          shape_invariants=[tf.TensorShape([None, None]), i.get_shape(),
                                                                            n.get_shape()])
                    fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=0)
                    if index == 0:
                        new_bot = fullsize_bottom
                    else:
                        new_bot = tf.concat([new_bot, fullsize_bottom], axis=0)

                new_bot = tf.expand_dims(new_bot, axis=3)
                imgpatch = output_high + new_bot

                loss, output, _ = trainlayer(imgpatch, gt, sess)

                setconf('ft', lev)
                summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(model_ckp, sess.graph)

                global_step = tf.Variable(0, name="global_step", trainable=False)

                variable_to_train = []
                for variable in tf.trainable_variables():
                    if not (variable.name.startswith(config.model.loss_model)):
                        variable_to_train.append(variable)
                train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step,
                                                                 var_list=variable_to_train)

                variables_to_restore = []
                for v in tf.global_variables():
                    if not (v.name.startswith(config.model.loss_model)):
                        variables_to_restore.append(v)
                saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                ''' restore high frequency vars '''
                variables_to_restore = []
                for v in tf.trainable_variables():
                    if v.name.startswith('high'):
                        variables_to_restore.append(v)

                setconf('high', lev)
                saver_h = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                ckpt = tf.train.get_checkpoint_state(model_ckp)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(model_ckp)
                    saver_h.restore(sess, full_path)

                ''' restore low frequency vars '''
                variables_to_restore = []
                for v in tf.trainable_variables():
                    if v.name.startswith('bot'):
                        variables_to_restore.append(v)

                setconf('bot', lev)
                saver_l = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                ckpt = tf.train.get_checkpoint_state(model_ckp)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(model_ckp)
                    saver_l.restore(sess, full_path)

                setconf('ft', lev)
                # restore variables for training model if the checkpoint file exists.
                epoch = restoreandgetepochs(model_ckp, sess, batchnum, saver)

                ####################
                """Start Training"""
                ####################
                start_time = time.time()
                while True:
                    _, loss_t, step, predict, gtruth = sess.run([train_op, loss, global_step, output, gt])
                    batch_id = int(step % batchnum)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    """logging"""
                    tf.logging.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, global step: %4d"
                                    % (epoch + 1, batch_id, batchnum, elapsed_time, loss_t, step))

                    # advance counters
                    if batch_id == 0:
                        if epoch >= goal_epoch:
                            break
                        else:
                            """checkpoint"""
                            saver.save(sess, os.path.join(model_ckp, 'pynets-model-ft.ckpt'), global_step=step)
                        epoch += 1

                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()


def trainlayer(imgpatch, gt, sess):
    ##################
    """Feed Network"""
    ##################
    output = ns.netftlayer(imgpatch)

    '''l2 loss'''
    loss_l2 = tf.reduce_mean((output - gt)**2)

    '''perceptual loss'''
    # duplicate the colour channel to be 3 same layers.
    output_3_channels = tf.concat([output, output, output], axis=3)
    gt_gray_3_channels = tf.concat([gt, gt, gt], axis=3)

    losses = cal_loss(output_3_channels, gt_gray_3_channels, config.model.loss_vgg, sess)
    loss_f = losses.loss_f / 3

    loss = loss_l2 * 0.6 + loss_f * 0.4

    #################
    """Add Summary"""
    #################
    tf.summary.scalar('loss/loss_l2', loss_l2 * 0.6)
    tf.summary.scalar('loss/loss_f', loss_f * 0.4)
    tf.summary.scalar('loss/total_loss', loss)
    tf.summary.image('input', imgpatch, max_outputs=12)
    tf.summary.image('output', output, max_outputs=12)
    tf.summary.image('ground_truth', gt, max_outputs=12)

    return loss, output, gt


def load(ckpt_dir, sess, saver):
    tf.logging.info('reading checkpoint')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        global_step = int(full_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, full_path)
        return True, global_step
    else:
        return False, 0


def restoreandgetepochs(ckpt_dir, sess, batchnum, savaer):
    status, global_step = load(ckpt_dir, sess, savaer)
    if status:
        start_epoch = global_step // batchnum
        tf.logging.info('model restore success')
    else:
        start_epoch = 0
        tf.logging.info("[*] Not find pretrained model!")
    return start_epoch


def calshape(h, w, lev):
    new_h, new_w = h, w
    for i in range(int(lev)):
        new_h = int(new_h / 2)
        new_w = int(new_w / 2)
    return (new_h, new_w)


main(level, epochs)

