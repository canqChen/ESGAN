import argparse
import os
import tensorflow as tf
from ESGAN import ESGAN
from utils import check_folder
# tf.set_random_seed(19)

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Implementation of ESGAN')

    parser.add_argument('--EID', dest='EID', type=int, default='1', help='# of experiment')
    parser.add_argument('--ref_A', dest='ref_A', type=str, default=1, help='# of reference image')
    parser.add_argument('--ref_B', dest='ref_B', type=str, default=1, help='# of reference image')

    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test, sg_test')
    parser.add_argument('--dataset_name', dest='dataset_name', default='joy2sadness', help='name of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')
    parser.add_argument('--decay_epoch', dest='decay_epoch', type=int, default=100, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
    # network properties
    parser.add_argument('--lambda_gan',dest='lambda_gan', type=float, default=1, help='weight of adversarial loss')
    parser.add_argument('--lambda_c', dest='lambda_c', type=float, default=5, help='weight on content loss term in objective')
    parser.add_argument('--lambda_e', dest='lambda_e', type=float, default=5, help='weight on emotion loss term in objective')
    parser.add_argument('--lambda_recon', dest='lambda_recon', type=float, default=10, help='weight on self-reconstruction loss term in objective')
    parser.add_argument('--lambda_cc_recon', dest='lambda_cc_recon', type=float, default=10, help='weight on cross-domain reconstruction loss term in objective')
    parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
    parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
    parser.add_argument('--use_deconv', dest='use_deconv', action='store_true', default=False, help='deconv use or not')
    parser.add_argument('--pool_size', dest='pool_size', type=int, default=30, help='max size of image pool, 0 means do not use image pool')
    parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
    parser.add_argument('--n_dis', dest='n_dis', type=int, default=5, help='number of discriminator layer')
    parser.add_argument('--n_scale', dest='n_scale',type=int, default=1, help='number of scales')
    parser.add_argument('--n_res', dest='n_res', type=int, default=5, help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--n_sample', dest='n_sample', type=int, default=2, help='number of sampling layers in content encoder')
    parser.add_argument('--augment_flag', dest='augment_flag', action='store_true', default=False, help='Image augmentation use or not')
    parser.add_argument('--use_content_perceptual', dest='use_content_perceptual', action='store_true', default=False, help='Perceptual content loss use or not')
    parser.add_argument('--use_gan_augment', dest='use_gan_augment', action='store_true', default=False, help='real images of opposite domain logit use or not')

    # image properties
    parser.add_argument('--img_ch', dest='img_ch', type=int, default=3, help='# of input image channels')
    parser.add_argument('--aug_size', dest='aug_size', type=int, default=30, help='scale images to size [w+aug_size,h+aug_size]')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256, help='then crop to this size')

    parser.add_argument('--save_freq', dest='save_freq', type=int, default=500, help='save a model every save_freq steps')
    parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=500, help='sample generated images every sample_freq iterations')
    # directories
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
    parser.add_argument('--logs_dir', dest='logs_dir', default='./logs', help='logs are saved here') 

    parser.add_argument('--visible_devices', dest='visible_devices', type=str, default=None, help='set cuda_visible_devices')
    # return check_args(parser.parse_args())
    return parser.parse_args()

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --test_dir
    check_folder(args.test_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def main(_):
    args = parse_args()
    if args.visible_devices != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = ESGAN(sess, args)
        if args.phase == 'train':
            model.train()
            print(' [*] Training finished!')
        elif args.phase=='test':
            model.test()
            print(' [*] Test finished!')
        elif args.phase=='sg_test':
            model.sample_guide_test()
            print(' [*] Example-guided test finished!')
        elif args.phase=='ref_test':
            model.ref_test(args)
            print(' [*] Reference test finished!')
        elif args.phase=='recon':
            model.reconstruct_image()
            print(' [*] Reconstructing images finished!')
        else:
            print(' [!] Phase error!')

if __name__ == '__main__':
    tf.app.run()
