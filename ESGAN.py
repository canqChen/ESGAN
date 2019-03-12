from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tensorflow.contrib.data import *

from utils import *
from ops import *


class ESGAN(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.model_name = "ESGAN"
        self.EID = args.EID

        self.dataset_name = args.dataset_name
        self.is_training = (args.phase=='train')

        self.epoch = args.epoch
        self.decay_epoch = args.decay_epoch
        self.batch_size = args.batch_size
        self.init_lr = args.lr
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.aug_size = args.aug_size

        self.lambda_gan = args.lambda_gan
        self.lambda_c = args.lambda_c
        self.lambda_e = args.lambda_e
        self.lambda_recon = args.lambda_recon
        self.lambda_cc_recon = args.lambda_cc_recon
        
        self.augment_flag = args.augment_flag
        self.use_content_perceptual = args.use_content_perceptual
        self.use_gan_augment = args.use_gan_augment
        self.save_freq = args.save_freq
        self.sample_freq = args.sample_freq

        self.n_res = args.n_res
        self.n_sample = args.n_sample
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.gf_ch = args.ngf
        self.df_ch = args.ndf
        
        self.sample_dir = check_folder(os.path.join(args.sample_dir, self.model_dir))
        self.checkpoint_dir = check_folder(os.path.join(args.checkpoint_dir, self.model_dir))
        self.test_dir = check_folder(os.path.join(args.test_dir, self.model_dir))
        self.logs_dir = check_folder(os.path.join(args.logs_dir, self.model_dir))

        self.use_deconv = args.use_deconv
        
        self.dataA = glob('../datasets/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.dataB = glob('../datasets/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.dataA), len(self.dataB))
        
        # image pool buffer, restore previously generated max_size iamges
        self.pool = ImagePool(args.pool_size)
        self._build_model()
        show_all_variables()
        self._show_network_arch()
        
    def _show_network_arch(self):
        print("##### Information #####")
        print("# experiment ID : ", self.EID)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# Employ data augmentation : ", self.augment_flag)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# decay epoch : ", self.decay_epoch)
        print("# iteration per epoch : ", self.dataset_num//self.batch_size)
        print("# initial learning rate : ", self.init_lr)
        print()
        print("##### Encoder #####")
        print("# Use perceptual content loss : ", self.use_content_perceptual)
        print("# Down sample : ", self.n_sample)
        print("# Residual blocks : ", self.n_res)
        print()
        print("##### Decoder #####")
        print("# Residual blocks : ", self.n_res)
        print("# Up sample : ", self.n_sample)
        print()
        print("##### Discriminator #####")
        print("# Use gan augmentation : ", self.use_gan_augment)
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)
        print()

    def content_encoder(self, image, reuse=True, scope="content_encoder"):
        channels = self.gf_ch
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            feats = []
            # image: 256 x 256 x img_ch

            output = tf.nn.relu(instance_norm(conv2d(image, channels, ks=7, s=1, pad=3 , scope='g_ce_c1'), 'g_ce_in1'))
            feats.append(output)
            # output: 256 x 256 x channels
            # down-sample
            channels *= 2
            for i in range(self.n_sample):
                with tf.variable_scope('downsample_%d'%(i+1)):
                    output = tf.nn.relu(instance_norm(conv2d(output, channels, 4, 2, pad=1, scope='g_ce_c%d'%(i+2)), 'g_ce_in%d'%(i+2)))
                    channels *= 2
                    feats.append(output)
            # output size: 256/2^n_sample x 256/2^n_sample x channels*2^n_sample
            channels = output.get_shape().as_list()[-1]
            for i in range(self.n_res):
                output = resblock(output, channels, scope='g_ce_res%d'%(i+1))
                feats.append(output)

            return output,feats

    def emotion_encoder(self, image, reuse=True, scope="emotion_encoder"):
        channels = self.gf_ch
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            feats = []
            # image: 256 x 256 x img_ch
            
            # output = tf.nn.relu(instance_norm(conv2d(image, channels, ks=7, s=1, pad=3, scope='g_ee_c1'), 'g_ee_in1'))
            output = tf.nn.relu(conv2d(image, channels, ks=7, s=1, pad=3, scope='g_ee_c1'))
            feats.append(output)

            # output: 256 x 256 x channels
            # down-sample
            channels *= 2
            for i in range(self.n_sample):
                with tf.variable_scope('downsample_%d'%(i+1)):
                    # output = tf.nn.relu(instance_norm(conv2d(output, channels, 4, 2, pad=1, scope='g_ee_c%d'%(i+2)), 'g_ee_in%d'%(i+2)))
                    output = tf.nn.relu(conv2d(output, channels, 4, 2, pad=1, scope='g_ee_c%d'%(i+2)))
                    feats.append(output)
                    channels *= 2

            channels = output.get_shape().as_list()[-1]
            for i in range(self.n_res):
                # output = tf.nn.relu(instance_norm(conv2d(output, channels, 3, 1, pad=1, scope='g_ee_c%d'%(self.n_sample+2+i)),'g_ee_in%d'%(self.n_sample+2+i)))
                output = tf.nn.relu(conv2d(output, channels, 3, 1, pad=1, scope='g_ee_c%d'%(self.n_sample+2+i)))
                # output = resblock(output, channels, scope='g_ee_res%d'%(i+1))
                feats.append(output)
            return feats

    def decoder(self, content_code, emotion_feats, reuse=True, scope="decoder"):
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            
            channels = content_code.get_shape().as_list()[-1]
            idx = len(emotion_feats) - 1
            output = content_code
            # n_res resdual blocks
            for i in range(self.n_res):
                output = adaptive_resblock(output, emotion_feats[idx], channels, scope='g_dc_ada_res%d'%(i+1))
                idx -= 1

            # for i in range(2*self.n_res-1):
            #     if i % 2 == 0:
            #         output = adaptive_resblock(output, emotion_feats[idx], channels, scope='g_dc_ada_res%d'%((i+2)/2))
            #         idx -= 1
            #     else:
            #         output = resblock(output, channels, scope='g_dc_res%d'%(i/2+1))
            # for i in range(self.n_res):
            #     output = adaptive_resblock(output, emotion_feats[idx], channels, scope='g_dc_ada_res%d'%(2*i+1))
            #     output = adaptive_resblock(output, emotion_feats[idx], channels, scope='g_dc_ada_res%d'%(2*i+2))
            #     idx -= 1
            # channels = channels//2
            output = tf.nn.relu(ada_instance_norm(conv2d(output, channels, 3, 1, pad=1, scope='g_dc_conv1'), emotion_feats[idx], scope='g_dc_adin'))
            idx -= 1

            channels = channels//2
            
            for i in range(self.n_sample):
                with tf.variable_scope('upsample_%d'%(i+1)):
                    if self.use_deconv:
                        output = tf.nn.relu(ada_instance_norm(deconv2d(output, channels, 3, 2, scope='g_dc_deconv'), emotion_feats[idx], scope='g_dc_adin'))
                    else:
                        output = tf.nn.relu(ada_instance_norm(nearest_upsample_conv(output, channels, 3, 1, pad=1, scope='g_dc_upsp_conv'), emotion_feats[idx], scope='g_dc_upsp_adin'))
                idx -= 1
                channels = channels//2

            output = tf.nn.relu(conv2d(output, channels*2, 3, 1, pad=1, use_bias=True, scope='g_dc_conv2'))

            pred = tf.nn.tanh(conv2d(output, self.img_ch, 7, 1, pad=3, use_bias=True, scope='g_dc_pred_conv'))

            return pred

    def discriminator(self, image, reuse=True, scope="discriminator"):
        with tf.variable_scope(scope):
            # image is 256 x 256 x img_ch
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            D_logit = []

            for scale in range(self.n_scale):
                channels = self.df_ch

                output = lrelu(conv2d(image, channels, use_bias=True, scope='d%d_conv1'%(scale+1)))

                channels *= 2
                for i in range(1,self.n_dis):
                    output = lrelu(instance_norm(conv2d(output, channels, scope='d%d_conv%d'%(scale+1,i+1)), 'd%d_in%d'%(scale+1,i+1)))
                    channels *= 2
                output = conv2d(output, 1, ks=1, s=1, scope='d%d_logit'%(scale+1))
                D_logit.append(output)

                image = down_sample(image)
            return D_logit

    def encode(self, image, reuse=True):
        content_code,content_feats = self.content_encoder(image, reuse=reuse)
        emotion_feats = self.emotion_encoder(image, reuse=reuse)
        return content_code, content_feats, emotion_feats

    def _build_model(self):

        Image_Data_Class = ImageData(self.img_size, self.img_size, self.img_ch, self.aug_size, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.dataA)
        trainB = tf.data.Dataset.from_tensor_slices(self.dataB)

        # tensorflow 1.8+
        trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).prefetch(self.batch_size)
        trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).prefetch(self.batch_size)
        
        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()

        self.real_A = trainA_iterator.get_next()
        self.real_B = trainB_iterator.get_next()

        # encode
        self.a_content_code, self.a_content_feats, self.a_emotion_feats = self.encode(self.real_A, reuse=False)
        self.b_content_code, self.b_content_feats, self.b_emotion_feats = self.encode(self.real_B)

        # within domain decode
        self.a_recon = self.decoder(self.a_content_code, self.a_emotion_feats, reuse=False)
        self.b_recon = self.decoder(self.b_content_code, self.b_emotion_feats)
        # cross domain decode
        self.fake_A = self.decoder(self.b_content_code, self.a_emotion_feats)
        self.fake_B = self.decoder(self.a_content_code, self.b_emotion_feats)
        
        # cross-domain encode
        self.b_recon_content_code, self.b_recon_content_feats, self.a_recon_emotion_feats = self.encode(self.fake_A)
        self.a_recon_content_code, self.a_recon_content_feats, self.b_recon_emotion_feats = self.encode(self.fake_B)

        # cross-domain reconstruction
        self.a_cc_recon = self.decoder(self.a_recon_content_code, self.a_recon_emotion_feats)
        self.b_cc_recon = self.decoder(self.b_recon_content_code, self.b_recon_emotion_feats)

        # discrimination of fake samples
        self.DB_fake = self.discriminator(self.fake_B, reuse=False, scope="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, reuse=False, scope="discriminatorA")

        # loss function
        # content loss
        if self.use_content_perceptual:
            self.a_content_loss = self.lambda_c * perceptual_loss(self.a_content_feats[-1::-2], self.a_recon_content_feats[-1::-2])
            self.b_content_loss = self.lambda_c * perceptual_loss(self.b_content_feats[-1::-2], self.b_recon_content_feats[-1::-2])
        else:
            self.a_content_loss = self.lambda_c * mae_loss(self.a_content_code, self.a_recon_content_code)
            self.b_content_loss = self.lambda_c * mae_loss(self.b_content_code, self.b_recon_content_code)

        # emotion loss
        self.a_emotion_loss = self.lambda_e * perceptual_loss(self.a_emotion_feats[-1::-2], self.a_recon_emotion_feats[-1::-2])
        self.b_emotion_loss = self.lambda_e * perceptual_loss(self.b_emotion_feats[-1::-2], self.b_recon_emotion_feats[-1::-2])

        # self-reconstruction loss
        self.a_recon_loss = self.lambda_recon * mae_loss(self.real_A, self.a_recon)
        self.b_recon_loss = self.lambda_recon * mae_loss(self.real_B, self.b_recon)
        # cross-domain reconstruction
        self.a_cc_recon_loss = self.lambda_cc_recon * mae_loss(self.real_A,self.a_cc_recon)
        self.b_cc_recon_loss = self.lambda_cc_recon * mae_loss(self.real_B,self.b_cc_recon)

        # define g loss, use lsgan
        self.g_ad_loss_a2b = self.lambda_gan * generator_loss(self.DB_fake)
        self.g_ad_loss_b2a = self.lambda_gan * generator_loss(self.DA_fake)

        self.g_loss = self.g_ad_loss_a2b + self.g_ad_loss_b2a + self.a_content_loss \
            + self.b_content_loss + self.a_emotion_loss + self.b_emotion_loss \
            + self.a_recon_loss + self.b_recon_loss + self.a_cc_recon_loss + self.b_cc_recon_loss

        # define d loss
        # feed the discriminator with generated image buffer
        self.fake_A_sample_buffer = tf.placeholder(tf.float32,
                                                   [None, self.img_size, self.img_size,
                                                    self.img_ch], name='fake_A_sample_buffer')
        self.fake_B_sample_buffer = tf.placeholder(tf.float32,
                                                   [None, self.img_size, self.img_size,
                                                    self.img_ch], name='fake_B_sample_buffer')

        self.DB_real = self.discriminator(self.real_B, scope="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, scope="discriminatorA")

        self.DB_fake_sample = self.discriminator(self.fake_B_sample_buffer, scope="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample_buffer, scope="discriminatorA")

        if self.use_gan_augment:
            self.DA_real_ = self.discriminator(self.real_A, scope="discriminatorB")
            self.DB_real_ = self.discriminator(self.real_B, scope="discriminatorA")
        else:
            self.DA_real_ = self.DB_real_=None

        self.db_loss = self.lambda_gan * discriminator_loss(self.DB_real, self.DB_fake_sample, self.DA_real_)
        self.da_loss = self.lambda_gan * discriminator_loss(self.DA_real, self.DA_fake_sample, self.DB_real_)
        self.d_loss = self.da_loss + self.db_loss

        # summary
        self.a_content_loss_sum = tf.summary.scalar("a_content_loss", self.a_content_loss)
        self.b_content_loss_sum = tf.summary.scalar("b_content_loss", self.b_content_loss)
        self.a_emotion_loss_sum = tf.summary.scalar("a_emotion_loss", self.a_emotion_loss)
        self.b_emotion_loss_sum = tf.summary.scalar("b_emotion_loss", self.b_emotion_loss)
        self.a_recon_loss_sum = tf.summary.scalar("a_recon_loss", self.a_recon_loss)
        self.b_recon_loss_sum = tf.summary.scalar("b_recon_loss", self.b_recon_loss)
        self.a_cc_recon_loss_sum = tf.summary.scalar("a_cc_recon_loss", self.a_cc_recon_loss)
        self.b_cc_recon_loss_sum = tf.summary.scalar("b_cc_recon_loss", self.b_cc_recon_loss)
        self.g_ad_loss_a2b_sum = tf.summary.scalar("g_ad_loss_a2b", self.g_ad_loss_a2b)
        self.g_ad_loss_b2a_sum = tf.summary.scalar("g_ad_loss_b2a", self.g_ad_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge(
            [self.a_content_loss_sum, self.b_content_loss_sum, self.a_emotion_loss_sum, self.b_emotion_loss_sum, 
            self.a_recon_loss_sum, self.b_recon_loss_sum, self.a_cc_recon_loss_sum, self.b_cc_recon_loss_sum, 
            self.g_ad_loss_a2b_sum,self.g_ad_loss_b2a_sum, self.g_loss_sum])

        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.d_sum = tf.summary.merge([self.da_loss_sum,self.db_loss_sum,self.d_loss_sum])

        # training params
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if ('encoder' in var.name) or ('decoder' in var.name)]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        # testing phase
        self.test_input_A = tf.placeholder(tf.float32,
                                           [None, self.img_size, self.img_size,
                                            self.img_ch], name='test_input_A')
        self.test_input_B = tf.placeholder(tf.float32,
                                           [None, self.img_size, self.img_size,
                                            self.img_ch], name='test_input_B')
        # encode
        a_content_code_test, _, a_emotion_feats_test = self.encode(self.test_input_A)
        b_content_code_test, _, b_emotion_feats_test = self.encode(self.test_input_B)
        # cross domain decode
        self.test_fake_A = self.decoder(b_content_code_test, a_emotion_feats_test)
        self.test_fake_B = self.decoder(a_content_code_test, b_emotion_feats_test)
        # self decode
        self.rec_A = self.decoder(a_content_code_test,a_emotion_feats_test)
        self.rec_B = self.decoder(b_content_code_test,b_emotion_feats_test)
        # encode again
        b_recon_content_code_test, _, a_recon_emotion_feats_test = self.encode(self.test_fake_A)
        a_recon_content_code_test, _, b_recon_emotion_feats_test = self.encode(self.test_fake_B)
        # cross cycle reconstruction
        self.cc_rec_A = self.decoder(a_recon_content_code_test,a_recon_emotion_feats_test)
        self.cc_rec_B = self.decoder(b_recon_content_code_test,b_recon_emotion_feats_test)

    def train(self):
        """Train esgan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # saver to save model
        self.saver = tf.train.Saver()
        # summary writer
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        
        steps = self.dataset_num // self.batch_size
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter + 1
            start_epoch = checkpoint_counter//steps + 1
            start_step = checkpoint_counter % steps + 1
            print(" [*] Successfully Loaded !")
        else:
            counter = 1
            start_epoch = 1
            start_step = 1
            print(" [!] No Checkpoint To Load...")

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch + 1):
            # decay learning rate
            # lr = self.init_lr if epoch <= self.decay_epoch else self.init_lr * (self.epoch-epoch + 1)/(self.epoch-self.decay_epoch)
            # lr = self.init_lr if epoch <= self.decay_epoch else 0.5*self.init_lr
            lr = self.init_lr
            for idx in range(start_step, steps+1):
                # Update G network and record fake outputs
                real_A, real_B, fake_A, fake_B, _, g_loss, g_summary_str = self.sess.run(
                    [self.real_A,self.real_B,self.fake_A, self.fake_B, self.g_optim, self.g_loss, self.g_sum],
                    feed_dict={self.lr: lr})
                
                # previously created images buffer,use to update the discriminator
                [fake_A_buffer, fake_B_buffer] = self.pool([fake_A, fake_B])

                # Update D network
                _, d_loss, d_summary_str = self.sess.run([self.d_optim, self.d_loss, self.d_sum],
                            feed_dict={self.fake_A_sample_buffer: fake_A_buffer,
                            self.fake_B_sample_buffer: fake_B_buffer, self.lr: lr})
                if counter % 20 == 0:
                    self.writer.add_summary(g_summary_str, counter)
                    self.writer.add_summary(d_summary_str, counter)

                # print informations after each training step
                print(("Epoch: [%3d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, steps, time.time() - start_time, d_loss, g_loss)))

                # sample generated results,and save them
                if counter % self.sample_freq == 0:
                    num_A = len(real_A)
                    integ_AB = []
                    integ_BA = []
                    for i in range(num_A):
                        integ_AB.append(real_A[i,:])
                        integ_AB.append(fake_B[i,:])
                        integ_BA.append(real_B[i,:])
                        integ_BA.append(fake_A[i,:])
                    integ_AB = np.array(integ_AB)
                    integ_BA = np.array(integ_BA)
                    save_images(integ_AB, [self.batch_size, 2], './{}/A2B_{:03d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))
                    save_images(integ_BA, [self.batch_size, 2], './{}/B2A_{:03d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))
                # save model each save_freq steps
                if counter % self.save_freq == 0:
                    self.save(self.checkpoint_dir, counter)
                counter += 1
            start_step = 1
            # save model
            self.save(self.checkpoint_dir, counter - 1)

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Loading Checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            return False, 0

    def test(self):
        """Test eegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        # load all the filenames of the images from the two datasets
        dataA = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testA'))
        dataB = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testB'))

        np.random.shuffle(dataA)
        np.random.shuffle(dataB)

        files_path_pairs = list(zip(dataA, dataB))
        A, B = self.dataset_name.split('2')

        # load model
        if self.load(self.checkpoint_dir):
            print(" [*] Successfully Loaded !")
        else:
            print(" [!] Load Failed...\n [!] Exit...")
            return None

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'test_result.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>{0}2{1}</th><th>source</th><th>target</th><th>{2}2{3}</th><th>source</th><th>target</th></tr>".format(A, B, B, A))

        for path_pair in files_path_pairs:
            print('Processing image: {0} and {1}'.format(path_pair[0], path_pair[1]))
            testA_img = [load_single_image(path_pair[0])]
            testA_img = np.array(testA_img).astype(np.float32)
            testB_img = [load_single_image(path_pair[1])]
            testB_img = np.array(testB_img).astype(np.float32)

            saveAtoB_path = os.path.join(self.test_dir, 'AtoB_{0}'.format(os.path.basename(path_pair[0])))
            saveBtoA_path = os.path.join(self.test_dir, 'BtoA_{0}'.format(os.path.basename(path_pair[1])))

            fakeB_img, fakeA_img = self.sess.run([self.test_fake_B, self.test_fake_A], 
                                feed_dict={self.test_input_A: testA_img, self.test_input_B: testB_img})

            save_images(fakeB_img, [1, 1], saveAtoB_path)
            save_images(fakeA_img, [1, 1], saveBtoA_path)
            # write results into html
            index.write("<tr>")
            index.write("<td>%s</td>" % os.path.basename(path_pair[0]))
            index.write("<td><img src='%s'></td>" % (path_pair[0] if os.path.isabs(path_pair[0]) else (
                '..' + os.path.sep + '..' + os.path.sep + path_pair[0])))
            index.write("<td><img src='%s'></td>" %
                        ('./' + os.path.basename(saveAtoB_path)))

            index.write("<td>%s</td>" % os.path.basename(path_pair[1]))
            index.write("<td><img src='%s'></td>" % (path_pair[1] if os.path.isabs(path_pair[1]) else (
                '..' + os.path.sep + '..' + os.path.sep + path_pair[1])))
            index.write("<td><img src='%s'></td>" %
                        ('./' + os.path.basename(saveBtoA_path)))

            index.write("</tr>")
        index.write("</table></body></html>")
        index.close()
    def ref_test(self,args):
        """Test eegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        # load all the filenames of the images from the two datasets
        dataA = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testA'))
        dataB = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testB'))

        ref_A = args.ref_A.split(',')
        ref_B = args.ref_B.split(',')
        
        files_path_pairs = list(zip(dataA, dataB))
        A, B = self.dataset_name.split('2')

        # load model
        if self.load(self.checkpoint_dir):
            print(" [*] Successfully Loaded !")
        else:
            print(" [!] Load Failed...\n [!] Exit...")
            return None

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'test_result.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>{0}2{1}</th><th>source</th><th>result</th><th>{2}2{3}</th><th>source</th><th>result</th></tr>".format(A, B, B, A))
        
        for path_pair in files_path_pairs:
            print('Processing image: {0} and {1}'.format(path_pair[0], path_pair[1]))

            index_A = np.random.choice(ref_A)
            index_B = np.random.choice(ref_B)
            ref_A_dir = '../datasets/{}/{}.jpg'.format(self.dataset_name + '/testA',index_A)
            ref_B_dir = '../datasets/{}/{}.jpg'.format(self.dataset_name + '/testB',index_B)
            ref_A_img = np.array([load_single_image(ref_A_dir)]).astype(np.float32)
            ref_B_img = np.array([load_single_image(ref_B_dir)]).astype(np.float32)

            testA_img = np.array([load_single_image(path_pair[0])]).astype(np.float32)
            testB_img = np.array([load_single_image(path_pair[1])]).astype(np.float32)

            saveAtoB_path = os.path.join(self.test_dir, 'AtoB_{0}'.format(os.path.basename(path_pair[0])))
            saveBtoA_path = os.path.join(self.test_dir, 'BtoA_{0}'.format(os.path.basename(path_pair[1])))

            fakeB_img = self.sess.run(self.test_fake_B, 
                                feed_dict={self.test_input_A: testA_img, self.test_input_B: ref_B_img})

            fakeA_img = self.sess.run(self.test_fake_A, 
                                feed_dict={self.test_input_A: ref_A_img, self.test_input_B: testB_img})

            save_images(fakeB_img, [1, 1], saveAtoB_path)
            save_images(fakeA_img, [1, 1], saveBtoA_path)
            # write results into html
            index.write("<tr>")
            index.write("<td>%s</td>" % os.path.basename(path_pair[0]))
            index.write("<td><img src='%s'></td>" % (path_pair[0] if os.path.isabs(path_pair[0]) else (
                '..' + os.path.sep + '..' + os.path.sep + path_pair[0])))
            index.write("<td><img src='%s'></td>" %
                        ('./' + os.path.basename(saveAtoB_path)))

            index.write("<td>%s</td>" % os.path.basename(path_pair[1]))
            index.write("<td><img src='%s'></td>" % (path_pair[1] if os.path.isabs(path_pair[1]) else (
                '..' + os.path.sep + '..' + os.path.sep + path_pair[1])))
            index.write("<td><img src='%s'></td>" %
                        ('./' + os.path.basename(saveBtoA_path)))

            index.write("</tr>")
        index.write("</table></body></html>")
        index.close()
    def sample_guide_test(self):
        """Test eegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        # load all the filenames of the images from the two datasets
        dataA = glob('../datasets/{}/*.*'.format(self.dataset_name + '/sample_guide_testA'))
        dataB = glob('../datasets/{}/*.*'.format(self.dataset_name + '/sample_guide_testB'))

        # load model
        if self.load(self.checkpoint_dir):
            print(" [*] Successfully Loaded !")
        else:
            print(" [!] Load Failed...\n [!] Exit...")
            return None

        A, B = self.dataset_name.split('2')
        for i in range(10):
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            num = min(len(dataA),len(dataB))
            files_path_pairs = list(zip(dataA[0:num], dataB[0:num]))
            check_folder(self.test_dir+'_sg')
            # write html for visual comparison
            index_path = os.path.join(self.test_dir+'_sg', 'sample_guide_test_result_%d.html'%(i))
            index = open(index_path, "w")
            index.write("<html><body><table><tr>")
            index.write("<th>{0}2{1}</th><th>source</th><th>target</th><th>result</th> <th>{2}2{3}</th><th>source</th><th>target</th><th>result</th></tr>".format(A, B, B, A))

            for path_pair in files_path_pairs:
                print('Processing images: {0} and {1}'.format(path_pair[0], path_pair[1]))
                testA_img = np.array([load_single_image(path_pair[0])]).astype(np.float32)
                testB_img = np.array([load_single_image(path_pair[1])]).astype(np.float32)

                saveAtoB_path = os.path.join(self.test_dir+'_sg', 'AtoB_{}to{}'.format(os.path.basename(path_pair[0]).split('.')[0],os.path.basename(path_pair[1])))
                saveBtoA_path = os.path.join(self.test_dir+'_sg', 'BtoA_{}to{}'.format(os.path.basename(path_pair[1]).split('.')[0],os.path.basename(path_pair[0])))

                fakeB_img, fakeA_img = self.sess.run([self.test_fake_B, self.test_fake_A], 
                                    feed_dict={self.test_input_A: testA_img, self.test_input_B: testB_img})

                save_images(fakeB_img, [1, 1], saveAtoB_path)
                save_images(fakeA_img, [1, 1], saveBtoA_path)
                # write results into html
                index.write("<tr>")
                index.write("<td>%s</td>" % os.path.basename(path_pair[0]))
                index.write("<td><img src='%s'></td>" % (path_pair[0] if os.path.isabs(path_pair[0]) else (
                    '..' + os.path.sep + '..' + os.path.sep + path_pair[0])))
                index.write("<td><img src='%s'></td>" % (path_pair[1] if os.path.isabs(path_pair[1]) else (
                    '..' + os.path.sep + '..' + os.path.sep + path_pair[1])))
                index.write("<td><img src='%s'></td>" %('./' + os.path.basename(saveAtoB_path)))

                index.write("<td>%s</td>" % os.path.basename(path_pair[1]))
                index.write("<td><img src='%s'></td>" % (path_pair[1] if os.path.isabs(path_pair[1]) else (
                    '..' + os.path.sep + '..' + os.path.sep + path_pair[1])))
                index.write("<td><img src='%s'></td>" % (path_pair[0] if os.path.isabs(path_pair[0]) else (
                    '..' + os.path.sep + '..' + os.path.sep + path_pair[0])))
                index.write("<td><img src='%s'></td>" %('./' + os.path.basename(saveBtoA_path)))

                index.write("</tr>")
            index.write("</table></body></html>")
            index.close()

    def reconstruct_image(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('../datasets/{}/*.*'.format(self.dataset_name + '/testB'))
        sample_files = zip(test_A_files,test_B_files)
        self.saver = tf.train.Saver()
        could_load, _ = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...\n [!] Exit...")
            return None
        test_dir_A = os.path.join(self.test_dir,'recon_image_A')
        check_folder(test_dir_A)
        test_dir_B = os.path.join(self.test_dir,'recon_image_B')
        check_folder(test_dir_B)
        rec_A_error = []
        rec_B_error = []
        cc_rec_A_error = []
        cc_rec_B_error = []
        index_path = os.path.join(self.test_dir, 'recon_result.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>A</th><th>source</th><th>result</th> <th>B</th><th>source</th><th>result</th></tr>")
        for img_file in sample_files:
            print('Processing images: {} and {}'.format(img_file[0],img_file[1]))
            sample_image_a = np.asarray([load_single_image(img_file[0], fine_size=self.img_size)])
            sample_image_b = np.asarray([load_single_image(img_file[1], fine_size=self.img_size)])
            recon_image_a_path = os.path.join(test_dir_A, 'recon_{0}'.format(os.path.basename(img_file[0])))
            cc_recon_image_a_path = os.path.join(test_dir_A, 'cc_recon_{0}'.format(os.path.basename(img_file[0])))
            recon_image_b_path = os.path.join(test_dir_B, 'recon_{0}'.format(os.path.basename(img_file[1])))
            cc_recon_image_b_path = os.path.join(test_dir_B, 'cc_recon_{0}'.format(os.path.basename(img_file[1])))

            self_recon_img_a,cc_recon_img_a,self_recon_img_b,cc_recon_img_b = self.sess.run([self.rec_A,self.cc_rec_A,self.rec_B,self.cc_rec_B],
                                                 feed_dict={self.test_input_A: sample_image_a,self.test_input_B: sample_image_b})
            save_images(self_recon_img_a, [1, 1], recon_image_a_path)
            save_images(self_recon_img_b, [1, 1], recon_image_b_path)
            save_images(cc_recon_img_a, [1, 1], cc_recon_image_a_path)
            save_images(cc_recon_img_b, [1, 1], cc_recon_image_b_path)

            index.write("<tr>")
            index.write("<td>%s</td>" % os.path.basename(img_file[0]))
            index.write("<td><img src='%s'></td>" % (img_file[0] if os.path.isabs(img_file[0]) else (
                    '..' + os.path.sep + '..' + os.path.sep + img_file[0])))
            index.write("<td><img src='%s'></td>" %('./recon_image_A/' + os.path.basename(recon_image_a_path)))

            index.write("<td>%s</td>" % os.path.basename(img_file[1]))
            index.write("<td><img src='%s'></td>" % (img_file[1] if os.path.isabs(img_file[1]) else (
                    '..' + os.path.sep + '..' + os.path.sep + img_file[1])))
            index.write("<td><img src='%s'></td>" %('./recon_image_B/' + os.path.basename(recon_image_b_path)))

            index.write("</tr>")

            img_a = imread(img_file[0])
            img_b = imread(img_file[1])
            recon_img_a = imread(recon_image_a_path)
            recon_img_b = imread(recon_image_b_path)
            cc_recon_imga = imread(cc_recon_image_a_path)
            cc_recon_imgb = imread(cc_recon_image_a_path)

            rec_error_a = np.mean(np.abs(img_a-recon_img_a))
            rec_error_b = np.mean(np.abs(img_b-recon_img_b))
            cc_rec_error_a = np.mean(np.abs(img_a-cc_recon_imga))
            cc_rec_error_b = np.mean(np.abs(img_b-cc_recon_imgb))
            rec_A_error.append(rec_error_a)
            rec_B_error.append(rec_error_b)
            cc_rec_A_error.append(cc_rec_error_a)
            cc_rec_B_error.append(cc_rec_error_b)

        index.write("</table></body></html>")
        index.close()
        print("Recon error per-pixel of A: %f",np.mean(rec_A_error))
        print("Recon error per-pixel of B: %f",np.mean(rec_B_error))
        print("Cc recon error per-pixel of A: %f",np.mean(cc_rec_A_error))
        print("Cc recon error per-pixel of B: %f",np.mean(cc_rec_B_error))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.EID)