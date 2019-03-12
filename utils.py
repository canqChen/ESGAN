"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division

import copy
import math
import os,random
import numpy as np
import scipy.misc
from tensorflow.contrib import slim
import tensorflow as tf



def get_stddev(x, k_h, k_w): return 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan

class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

class ImageData:
    def __init__(self, img_h, img_w, channels, aug_size=30, augment_flag=False):
        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels
        self.augment_flag = augment_flag
        self.aug_size=aug_size

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        if self.augment_flag:
            augment_size_h = self.img_h + self.aug_size
            augment_size_w = self.img_w + self.aug_size
            p = random.random()
            if p > 0.5:
                img = self.augmentation(img, augment_size_h, augment_size_w)
        return img
    def augmentation(self, image, aug_img_h, aug_img_w):
        seed = random.randint(0, 2 ** 31 - 1)
        ori_image_shape = tf.shape(image)
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.resize_images(image, [aug_img_h, aug_img_w])
        image = tf.random_crop(image, ori_image_shape, seed=seed)
        return image

def load_single_image(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img

def load_image_pair(image_path, load_size=286, fine_size=256, is_testing=False):
    '''
    If is_testing==False, execute data augmentation
    return image pair AB with shape (fine_size, fine_size, input_c_dim + output_c_dim)
    '''
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        # random crop
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]
        # augmentation
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    # normalize images both in training and testing phase
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


# -----------------------------

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


def merge_images(images):
    return inverse_transform(images)

def merge(images, size):
    """merge all the images within a batch and arrage them in size[0] rows and size[1] columns

    Parameters
    ----------
    images: batch images
    size: [rows,columns]

    Returns
    -------
    return the integrated image
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]  #
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w = None,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
        # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    '''
    All the pixel values in images lie in [-1,1] because of the final activation function of tanh.
    Then transform every pixel value to [0,1]
    '''
    return (images+1.)/2.

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir