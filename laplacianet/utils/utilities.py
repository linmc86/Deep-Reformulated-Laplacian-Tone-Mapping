import tensorflow as tf
import scipy.misc
import cv2, os, sys, random
import numpy as np


def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img


def norm_0_to_255(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = ((img - min_value) * 255) / (max_value - min_value)
    return new_img


def tensor_norm_0_to_255(tensor_img):
    tensor_img = tf.to_float(tensor_img)

    tensor_img = tf.div(
        tf.subtract(
            tensor_img,
            tf.reduce_min(tensor_img)
        ) * 255.0,
        tf.subtract(
            tf.reduce_max(tensor_img),
            tf.reduce_min(tensor_img)
        )
    )
    return tensor_img


def convtfRGBtoBGR(rgb):
    red, green, blue = tf.split(rgb, 3, 3)
    bgr = tf.concat([blue, green, red], 3)
    return bgr


def convtfBGRtoRGB(bgr):
    blue, green, red = tf.split(bgr, 3, 3)
    bgr = tf.concat([red, green, blue], 3)
    return bgr


def eval_convtfBGRtoRGB(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_images_from_event(event_path, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(event_path):
            for v in e.summary.value:
                if v.tag.startswith(tag):
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    sys.stdout.flush()
                    scipy.misc.imsave(output_fn, im)
                    count += 1
    sess.close()


####lum2rgb
def lum(img):
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def lum2rgb(out_lum, in_lum, hdr_image):
    rgb = np.zeros(np.shape(hdr_image))
    rgb[:,:,0] = ((hdr_image[:,:,0]/(in_lum + 1e-10)) ** 0.6)*out_lum
    rgb[:,:,1] = ((hdr_image[:,:,1]/(in_lum + 1e-10)) ** 0.6)*out_lum
    rgb[:,:,2] = ((hdr_image[:,:,2]/(in_lum + 1e-10)) ** 0.6)*out_lum
    return rgb


def demosaicAndSaveImage(pic, y, s):
    pic = np.float32(pic)
    y = np.float32(y)
    lum = 0.2126 * pic[:,:,0] + 0.7152 * pic[:,:,1] + 0.0722 * pic[:,:,2]
    # s = 0.5
    # Demosaicing
    # demosaic_y = pic
    demosaic_y = np.zeros(np.shape(pic))
    demosaic_y[:,:,0] = ((pic[:,:,0]/(lum + 1e-10)) ** s)*y
    demosaic_y[:,:,1] = ((pic[:,:,1]/(lum + 1e-10)) ** s)*y
    demosaic_y[:,:,2] = ((pic[:,:,2]/(lum + 1e-10)) ** s)*y

    return demosaic_y


def cut_dark_end_percent(img, npercent):
    [h, w] = np.shape(img)
    num_elements = h * w
    cut_num = int(num_elements * npercent)
    flat_img = img.flatten()
    smallend = np.partition(flat_img, cut_num)[cut_num]
    largeend = np.partition(flat_img, -cut_num)[-cut_num]
    return np.clip(img, a_min=smallend, a_max=largeend)











