# coding: utf-8
from __future__ import print_function
from __future__ import division
from data_parser.parse_tfrec import *
from loss.cal_loss import *
from utils.utilities import *
from utils.configs import *
import net.net_new_structure as ns
import time, os, sys
import matplotlib.pyplot as plt

'''Parameters to modify'''
level = '4'
epochs = 50
'''===================='''


batchnum = config.train.batchnum_high
tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(lev, goal_epoch):
    model_ckp = config.model.ckp_path_high
    tfrecord_path = config.model.tfrecord_dual + lev + '_' + config.model.tfrecord_suffix

    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

                loss, output, gt = trainlayer(tfrecord_path, sess)

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
                    if batch_id == batchnum - 1:
                        if epoch >= goal_epoch:
                            break
                        else:
                            """checkpoint"""
                            saver.save(sess, os.path.join(model_ckp, 'pynets-model-high.ckpt'), global_step=step)
                        epoch += 1

                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()


def trainlayer(tfrecord_path, sess):
    """Read tfrecord"""
    train_iter = data_iterator_new_high(tfrecord_path)
    img_gray, gt_gray = train_iter.get_next()

    """Feed Network"""
    output = ns.nethighlayer(img_gray)

    """Build Losses"""
    loss_l1_reg = 0
    loss_l1 = tf.reduce_mean(tf.abs(output - gt_gray))

    # Calculate L2 Regularization value based on trainable weights in the network:
    weight_size = 0
    for variable in tf.trainable_variables():
        if not (variable.name.startswith(config.model.loss_model)):
            loss_l1_reg += tf.reduce_sum(tf.abs(variable)) * 2
            weight_size += tf.size(variable)
    loss_l2_reg = loss_l1_reg / tf.to_float(weight_size)

    '''perceptual loss'''
    # duplicate the colour channel to be 3 same layers.
    output_3_channels = tf.concat([output, output, output], axis=3)
    gt_gray_3_channels = tf.concat([gt_gray, gt_gray, gt_gray], axis=3)

    losses = cal_loss(output_3_channels, gt_gray_3_channels, config.model.loss_vgg, sess)
    loss_f = losses.loss_f / 3

    loss = loss_f * 0.5 + loss_l1 * 0.5 + loss_l2_reg * 0.2

    #################
    """Add Summary"""
    #################
    tf.summary.scalar('loss/loss_l1', loss_l1 * 0.5)
    tf.summary.scalar('loss/loss_l2_reg', loss_l2_reg * 0.2)
    tf.summary.scalar('loss/loss_f', loss_f * 0.5)
    tf.summary.scalar('loss/total_loss', loss)
    tf.summary.image('input', img_gray, max_outputs=12)
    tf.summary.image('output', output, max_outputs=12)
    tf.summary.image('ground_truth', gt_gray, max_outputs=12)

    return loss, output, gt_gray


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


main(level, epochs)

