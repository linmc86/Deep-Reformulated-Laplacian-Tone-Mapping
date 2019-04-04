import tensorflow as tf


def nethighlayer(hdr):
    numc = 32
    with tf.variable_scope(name_or_scope="high_level"):
        network = bn(conv_relu(net=hdr, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = conv1x1(network, 1, name='conv5_1x1', padding='SAME')
        output = hdr + network
    return output


def netbotlayer(hdr):
    numc = 32
    with tf.variable_scope(name_or_scope="bot_level"):
        network = bn(conv_relu(net=hdr, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = conv1x1(network, 1, name='conv5_1x1', padding='SAME')
        output = hdr + network
    return output


def netftlayer(hdr):
    numc = 32
    with tf.variable_scope(name_or_scope="ft_merger"):
        res = bn(conv_relu(net=hdr, in_c=1, out_c=numc, padding='SAME', name='conv_expand'), name='conv_bn')
        skp_1 = res
        res = bn(conv_relu(res, in_c=numc,out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c1'),name='c1_bn')
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c2'),name='c2_bn')
        res = res + skp_1

        skp_2 = res
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c3'),name='c3_bn')
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c4'),name='c4_bn')
        res = res + skp_2

        skp_3 = res
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c5'),name='c5_bn')
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c6'),name='c6_bn')
        res = res + skp_3

        skp_4 = res
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c7'),name='c7_bn')
        res = bn(conv_relu(res, in_c=numc, out_c=numc, w_size=3, strides=1, padding='SAME', name='res_c8'),name='c8_bn')
        res = res + skp_4

        output = conv1x1(res, 1, name='conv_1x1', padding='SAME')

    return output


def residual_block(net, in_c, out_c, w_size=3, strides=1, padding='SAME', name='res'):
    res = bn(conv_relu(net, in_c=in_c, out_c=out_c, w_size=w_size, strides=strides, padding=padding, name=name + '_c1'))
    res = bn(conv_relu(res, in_c=in_c, out_c=out_c, w_size=w_size, strides=strides, padding=padding, name=name + '_c2'))
    return res + net


def bn(inputs, epsilon=0.01, name='batch_norm'):
    inputs_shape = inputs.get_shape()
    mean, variance = tf.nn.moments(inputs, range(len(inputs_shape.as_list()) - 1))

    output = tf.nn.batch_normalization(inputs, mean, variance, None, None, variance_epsilon=epsilon, name=name)
    return output


def weight_variable(shape, name=None, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv_relu(net, in_c, out_c, name, w_size=3, strides=1, padding='VALID'):
    w = weight_variable([w_size, w_size, in_c, out_c], name=name)
    b = bias_variable([out_c], name=name)
    network = tf.nn.conv2d(input=net,
                           filter=w,
                           padding=padding,
                           strides=[1, strides, strides, 1],
                           name="{}_conv".format(name),
                           )
    network = tf.nn.leaky_relu(network + b, name="{}_relu".format(name))
    return network


def conv1x1(net, numfilters, name, padding='VALID'):
    return tf.layers.conv2d(net,
                            filters=numfilters,
                            strides=(1, 1),
                            kernel_size=(1, 1),
                            name="{}_conv1x1".format(name), padding=padding)
