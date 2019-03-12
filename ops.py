"""
Most codes from https://github.com/XHUJOY/CycleGAN-tensorflow and https://github.com/taki0112/StarGAN-Tensorflow
"""
import math
import numpy as np
import tensorflow as tf



weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None

def batch_norm(x, scope="bn"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, \
    updates_collections=None, epsilon=1e-5, scale=True, scope=scope)

def instance_norm(x, scope="in"):
    # with tf.variable_scope(name):
    #     depth = input.get_shape()[3]
    #     scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    #     offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    #     mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    #     epsilon = 1e-5
    #     inv = tf.rsqrt(variance + epsilon)
    #     normalized = (input-mean)*inv
    #     return scale*normalized + offset
    return tf.contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)

def layer_norm(x, scope='ln') :
    return tf.contrib.layers.layer_norm(x,center=True, scale=True,scope=scope)

def ada_instance_norm(content, emotion, epsilon=1e-5, scope='adain'):
    with tf.variable_scope(scope):
        meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
        meanS, varS = tf.nn.moments(emotion, [1, 2], keep_dims=True)
        sigmaC = tf.sqrt(tf.add(varC, epsilon))
        sigmaS = tf.sqrt(tf.add(varS, epsilon))
        return (content - meanC) * sigmaS / sigmaC + meanS

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5, scope='adain'):
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    return gamma * ((content - c_mean) / c_std) + beta

def AdaIN(content_features, style_features, alpha=1, epsilon = 1e-5):
    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keep_dims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keep_dims=True)

    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean,
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features


def conv2d(x, channels, ks=4, s=2, pad=1, pad_type='reflect', use_bias=False, scope="conv2d"):
    with tf.variable_scope(scope):
        if pad>0:
            if pad_type == 'zero' :
                x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        return tf.layers.conv2d(x, channels, ks, strides=s, kernel_initializer=weight_init,padding='VALID',
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)

def deconv2d(x, channels, ks=4, s=2, use_bias=False, scope="deconv2d"):
    with tf.variable_scope(scope):
        return tf.layers.conv2d_transpose(inputs=x, filters=channels,kernel_size=ks, 
                                        kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                        strides=s, padding='SAME', use_bias=use_bias)

# Residual-block
def resblock(x, channels, ks=3, s=1, pad_type='reflect', use_bias=False, scope='resblock'):
    # w = x.get_shape()[1]
    # p = int((w*(stride-1)+kernel_size-stride)/2)
    p = int((ks-1)/2)
    with tf.variable_scope(scope):
        y = conv2d(x, channels, ks=ks, s=s, pad=p, pad_type=pad_type, use_bias=use_bias, scope='conv1')
        y = instance_norm(y, scope='in1')
        y = relu(y)

        y = conv2d(y, channels, ks=ks, s=s,pad=p, pad_type=pad_type, use_bias=use_bias, scope='conv2')
        y = instance_norm(y, scope='in2')
    return x + y

def adaptive_resblock(x, emotion_feat, channels, ks=3, s=1, pad_type='reflect', use_bias=False, scope='adaptive_resblock'):
    p = int((ks-1)/2)
    with tf.variable_scope(scope):
        mu, sigma = tf.nn.moments(emotion_feat, [1, 2], keep_dims=True)

        y = conv2d(x, channels, ks=ks, s=s, pad=p, pad_type=pad_type, use_bias=use_bias, scope='conv1')
        y = adaptive_instance_norm(y, sigma, mu, scope='adain1')
        y = relu(y)

        y = conv2d(y, channels, ks=ks, s=s, pad=p, pad_type=pad_type, use_bias=use_bias, scope='conv2')
        y = adaptive_instance_norm(y, sigma, mu, scope='adain2')

        return y + x

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def nearest_upsample_conv(x, output_channels, ks=3, s=1, pad=0, scale_factor=2, scope='upsp_conv'):
    with tf.variable_scope(scope):
        _, h, w, _ = x.get_shape().as_list()
        new_size = [h * scale_factor, w * scale_factor]
        x = tf.image.resize_nearest_neighbor(x, size=new_size)
        return conv2d(x, output_channels, ks=ks, s=1, pad=pad)

# def gram_matrix(features):
#     # features should be a layer activation for single input,
#     # so the shape should be [1, height, width, channels]
#     shape = features.get_shape().as_list()
#     features = tf.reshape(features, [shape[1] * shape[2], shape[3]])
#     unnormalized_gram_m = tf.matmul(features, features, transpose_a=True)
#     return tf.div(unnormalized_gram_m, shape[1] * shape[2] * shape[3])

# def gram(layer):
#     shape = tf.shape(layer)
#     num_images = shape[0]
#     num_filters = shape[3]
#     size = tf.size(layer)
#     filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
#     grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size / FLAGS.BATCH_SIZE)
#     return grams

# def gram_matrix(activations):
#     height = tf.shape(activations)[1]
#     width = tf.shape(activations)[2]
#     num_channels = tf.shape(activations)[3]
#     gram_matrix = tf.transpose(activations, [0, 3, 1, 2]) 
#     gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
#     gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
#     return gram_matrix

'''
# pytorch version
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
'''

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x, name='leaky_relu')

def relu(x):
    return tf.nn.relu(x, name='relu')

def tanh(x):
    return tf.tanh(x, name='tanh')

def linear(input_, output_size, scope=None, stddev=0.02, bias_init_value=0.0, with_w=False):
    '''
        fully connected layer multiplication
    '''
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_init_value))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
            
def flatten(x) :
    return tf.layers.flatten(x)


def mae_loss(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mse_loss(in_, target):
    return tf.reduce_mean((in_-target)**2)

def bce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def discriminator_loss(real_logit, fake_logit, fake_logit_=None):
    n_scale = len(real_logit)
    loss = []
    real_loss = 0
    fake_loss = 0
    for i in range(n_scale):
        real_loss = tf.reduce_mean(tf.squared_difference(real_logit[i], 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit[i]))
        if fake_logit_ != None:
            fake_loss_ = tf.reduce_mean(tf.square(fake_logit_[i]))
            loss.append(real_loss + fake_loss + fake_loss_)
        else:
            loss.append(real_loss + fake_loss)
    return sum(loss)

def generator_loss(fake_logit):
    n_scale = len(fake_logit)
    loss = []
    fake_loss = 0
    for i in range(n_scale) :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logit[i], 1.0))
        loss.append(fake_loss)
    return sum(loss)

def perceptual_loss(emotion_feats, recon_emotion_feats):
    n_feat = len(emotion_feats)
    loss = []
    for i in range(n_feat):
        loss.append(tf.reduce_mean(tf.abs(emotion_feats[i]-recon_emotion_feats[i])))
    return sum(loss)