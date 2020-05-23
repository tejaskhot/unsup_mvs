from __future__ import print_function

import os
import time
import sys
import math
import argparse

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *

FLAGS = tf.app.flags.FLAGS

# params for datasets
tf.app.flags.DEFINE_string('dtu_data_root', '/home/tejas/unsup_mvs/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('log_dir', '/home/tejas/unsup_mvs/logs/lambda1_128_nc3',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('save_dir', '/home/tejas/unsup_mvs/saved_models/lambda1_128_nc3',
                           """Path to save the model checkpoints.""")
tf.app.flags.DEFINE_string('save_op_dir', '/home/tejas/unsup_mvs/saved_outputs/lambda1_128_nc3',
                            """Path to dir where outputs are dumped""")
tf.app.flags.DEFINE_integer('save_op_interval', 1,
                            """Interval to dump out outputs""")

tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('val_interval', 400,
                          """number of train steps after which to run 40 val steps""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('view_num', 7,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth steps when training.""")
tf.app.flags.DEFINE_integer('max_w', 640,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512,
                            """Maximum image height when train  ing.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('base_image_size', 128,
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_float('interval_scale', 1.6,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """training batch size""")
tf.app.flags.DEFINE_integer('epoch', 8,
                            """training epoch""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('smooth_lambda', 1.0,
                          """lamda weighting of image gradient in smooth loss""")

# # params for config
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path',
                            "",
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step',0,
                            """ckpt step.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """Flag to training model""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 10000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0

    def __iter__(self):
        while True:
            for data in self.sample_list:
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                for view in range(self.view_num):
                    image = center_image(cv2.imread(data[2 * view]))
                    cam = load_cam(open(data[2 * view + 1]))
                    cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * (cams[0][1, 3, 1])
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class DUMVS():

    def __init__(self, training_list, validation_list=None):

        self.config = tf.ConfigProto(allow_soft_placement = True)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        self.training_sample_size = len(training_list)
        print('sample number: ', self.training_sample_size)
        self.training_list = training_list
        self.do_val = False
        if(validation_list is not None):
            self.do_val =True
            self.validation_list = validation_list

        self.op_dir = os.path.abspath(FLAGS.save_op_dir)

        try:
            os.makedirs(self.op_dir)
        except:
            pass

        # training generators
        self.training_generator = iter(MVSGenerator(self.training_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32)
        # dataset from generator
        self.training_set = tf.data.Dataset.from_generator(lambda: self.training_generator, generator_data_type)
        self.training_set = self.training_set.batch(FLAGS.batch_size)
        self.training_iterator = self.training_set.make_initializable_iterator()
        self.next_train_tuple = self.training_iterator.get_next()
        #######VALIDATION##############

        if(self.do_val):
            self.val_generator = iter(MVSGenerator(self.validation_list, FLAGS.view_num))
            generator_data_type = (tf.float32, tf.float32, tf.float32)
            # dataset from generator
            self.val_set = tf.data.Dataset.from_generator(lambda: self.training_generator, generator_data_type)
            self.val_set = self.val_set.batch(FLAGS.batch_size)
            self.val_iterator = self.val_set.make_initializable_iterator()
            self.next_val_tuple = self.val_iterator.get_next()

        ########## optimization options ##########
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr_op = tf.train.exponential_decay(1.5*FLAGS.base_lr, global_step=self.global_step,
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        self.opt= tf.train.AdamOptimizer(FLAGS.base_lr, 0.95)


        tower_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    self.images = tf.placeholder(tf.float32)
                    self.cams = tf.placeholder(tf.float32)
                    self.depth_image = tf.placeholder(tf.float32)
                    images, cams, depth_image = self.images,self.cams,self.depth_image

                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    depth_interval = depth_interval

                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    # inference##########################################################
                    inf_images= tf.slice(images, [0, 0, 0, 0, 0], [-1, 3, -1, -1, 3])
                    depth_map, prob_map,ref_tower,view_towers = inference_3view(inf_images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                    # refinement
                    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                    refined_depth_map = depth_refine(
                        depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                    # loss
                    loss0, less_one_temp, less_three_temp = mvsnet_loss(
                        depth_map, depth_image, depth_interval)
                    loss1, self.less_one_accuracy, self.less_three_accuracy = mvsnet_loss(
                        refined_depth_map, depth_image, depth_interval)
                    self.loss = (loss0 + loss1) / 2

                    resized_imgs = tf.image.resize_bilinear(images[0,:,:,:,:], [int((FLAGS.max_h)/4), int((FLAGS.max_w)/4)],align_corners=True)
                    ref_resized = tf.slice(resized_imgs, [0, 0, 0, 0], [1, -1, -1, 3])
                    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [1, 1, 2, 4, 4]), axis=1)
                    dmap_coarse  = tf.squeeze(depth_map)
                    dmap = tf.squeeze(tf.slice(refined_depth_map,[0,0,0,0],[1,-1,-1,1]))

                    depth_end = depth_start + (FLAGS.max_d - 2) * depth_interval

                    warped = []
                    warped_gt_view = []

                    masks = []
                    self.reconstr_loss = 0
                    self.ssim_loss = 0
                    self.smooth_loss = 0
                    self.reconstr_tower_loss =0

                    reprojection_losses = []
                    grad_lossesx = []
                    grad_lossesy = []
                    self.K =  tf.slice(ref_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
                    for view in range(1, FLAGS.view_num): # for each non-ref view
                        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [1, 1, 2, 4, 4]), axis=1)
                        view_img = tf.slice(resized_imgs, [view, 0, 0, 0], [1, -1, -1, 3])
                        # warp view_img to the ref_img using the dmap of the ref_img
                        warped_view,mask = inverse_warping(view_img, ref_cam, view_cam, dmap)
                        warped.append(warped_view)
                        masks.append(mask)

                        recon_loss = compute_reconstr_loss_map(warped_view,ref_resized,mask,simple=0)
                        # replace all 0 values with INF
                        valid_mask = 1 - mask

                        reprojection_losses.append(recon_loss + 1e4*valid_mask)

                        #SSIM loss##
                        ##https: // github.com / tensorflow / models / blob / master / research / vid2depth / model.py  # L211##
                        if(view<3):
                            self.ssim_loss += tf.reduce_mean(ssim(ref_resized, warped_view, mask))

                    ##smooth loss##
                    ##https://github.com/tinghuiz/SfMLearner/blob/master/SfMLearner.py#L156 ##
                    self.smooth_loss += depth_smoothness(tf.expand_dims(tf.expand_dims(dmap, 0), -1), ref_resized, FLAGS.smooth_lambda)

                    # top-k operates along the last dimension, so swap the axes accordingly
                    reprojection_volume = tf.transpose(tf.stack(reprojection_losses), [1,2,3,4,0])
                    # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
                    top_vals, self.top_inds = tf.nn.top_k(tf.negative(reprojection_volume), k=3, sorted=False)
                    top_vals = tf.negative(top_vals)
                    top_mask = tf.less(top_vals,1e4*tf.ones_like(top_vals)) ##0 where top_vals >= 1e4
                    top_vals = tf.multiply(top_vals, tf.cast(top_mask, top_vals.dtype))

                    self.error_map = tf.reduce_sum(top_vals,-1)
                    self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(top_vals,-1))
                    self.warped = tf.stack(warped)
                    self.masks = tf.stack(masks)
                    self.dmap = dmap
                    self.dmap_coarse = dmap_coarse
                    self.resized_imgs = resized_imgs

                    # retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    self.unsup_loss =  12*self.reconstr_loss +  6*self.ssim_loss + 0.18*self.smooth_loss
                    grads = self.opt.compute_gradients(self.unsup_loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)
        
        # average gradient
        grads = average_gradients(tower_grads)
        # training opt
        self.train_opt = self.opt.apply_gradients(grads, global_step=self.global_step)

        # summary
        summaries.append(tf.summary.scalar('loss', self.loss))
        summaries.append(tf.summary.scalar('unsup_loss', self.unsup_loss))
        summaries.append(tf.summary.scalar('less_one_accuracy', self.less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_accuracy', self.less_three_accuracy))
        summaries.append(tf.summary.scalar('reconstr_loss', self.reconstr_loss))
        summaries.append(tf.summary.scalar('self.ssim_loss', self.ssim_loss))
        summaries.append(tf.summary.scalar('self.smooth_loss', self.smooth_loss))
        summaries.append(tf.summary.scalar('lr', self.lr_op))
        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.summary_op = tf.summary.merge(summaries)
        self.init_op = tf.global_variables_initializer()

        self.train_vars = tf.trainable_variables()
        vars_3d = [v for v in self.train_vars if v.name.startswith('3dconv')]
        vars_refine = [v for v in self.train_vars if v.name.startswith('refine')]
        gs = [v for v in self.train_vars if v.name.startswith('global_step')]
        self.unsup_vars = vars_3d + vars_refine + gs
        self.init_unsup = tf.variables_initializer(self.unsup_vars)



    def train_dumvs(self):

        with self.sess as sess:
            # initialization
            total_step = 0
            sess.run(self.init_op) #initialize
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            val_summary_path = os.path.join(os.path.dirname(FLAGS.log_dir),(os.path.basename(FLAGS.log_dir)+"_val"))
            val_summary_writer = tf.summary.FileWriter(val_summary_path, sess.graph)

            # load pre-trained model
            if len(FLAGS.pretrained_model_ckpt_path):
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step
            # training several epochs
            for epoch in range(FLAGS.epoch):
                # training of one epoch
                step = 0
                sess.run(self.training_iterator.initializer)
                sess.run(self.val_iterator.initializer)
                for _ in range(int(self.training_sample_size / FLAGS.num_gpus)):

                    # run one batch
                    start_time = time.time()
                    try:
                        images, cams, depth_image = sess.run(self.next_train_tuple)
                        feed_dict = {self.images: images, self.cams: cams, self.depth_image: depth_image}

                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three,res_img,dmap_out,dmap_coarse_out,\
                        dmap_gt,warped_out,m_out,rloss, ssim_loss_out, smooth_loss_out,photo_error_map,K_out = sess.run(
                            [self.summary_op, self.train_opt, self.loss, self.less_one_accuracy, self.less_three_accuracy,
                             self.resized_imgs,self.dmap,self.dmap_coarse,self.depth_image,self.warped,self.masks,
                             self.reconstr_loss, self.ssim_loss, self.smooth_loss,self.error_map,self.K],feed_dict=feed_dict)

                        ###  VALIDATION ###
                        if step % FLAGS.val_interval == 0:
                            val_loss=[]
                            val_loss_one =[]
                            val_loss_three =[]
                            val_recons_loss =[]

                            val_iters =40
                            for val_cnt in range(val_iters):
                                images, cams, depth_image = sess.run(self.next_val_tuple)
                                feed_dict = {self.images: images, self.cams: cams, self.depth_image: depth_image}

                                out_summary_op, out_loss, out_less_one, out_less_three, res_img, dmap_out, \
                                dmap_gt, warped_out, m_out, rloss, ssim_loss_out, smooth_loss_out,top_inds_val = sess.run(
                                    [self.summary_op,self.loss, self.less_one_accuracy,
                                     self.less_three_accuracy,
                                     self.resized_imgs, self.dmap, self.depth_image, self.warped,
                                     self.masks,
                                     self.reconstr_loss, self.ssim_loss, self.smooth_loss,self.top_inds], feed_dict=feed_dict)
                                print(Notify.INFO,
                                      'VALIDATION: step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f' %
                                      (step, total_step, out_loss, out_less_one, out_less_three),
                                      Notify.ENDC)
                                print(Notify.INFO,
                                      'VALIDATION reconstr loss (ploss with mask), %.4f,' % (rloss), Notify.ENDC)

                                val_loss.append(out_loss)
                                val_loss_one.append(out_less_one)
                                val_loss_three.append(out_less_three)
                                val_recons_loss.append(rloss)

                            val_summary = tf.Summary()
                            print(Notify.INFO,
                                  '\nMEAN VALIDATION LOSS: step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f\n' %
                                  (step, total_step, np.mean(val_loss), np.mean(val_loss_one), np.mean(val_loss_three)))
                            prefix = 'val_'
                            val_summary.value.add(tag="%sloss" % prefix, simple_value=np.mean(val_loss))
                            val_summary.value.add(tag="%sless_one" % prefix, simple_value=np.mean(val_loss_one))
                            val_summary.value.add(tag="%sless_three" % prefix, simple_value=np.mean(val_loss_three))
                            val_summary.value.add(tag="%srecon_loss" % prefix, simple_value=np.mean(val_recons_loss))
                            val_summary_writer.add_summary(val_summary, total_step)
                        ### END VALIDATION ###


                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                            'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                            (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)

                        print(Notify.INFO,
                            'reconstr loss (ploss with mask), %.4f,' % (rloss), Notify.ENDC)


                        print(Notify.INFO,
                            'ssim loss: %.4f,  smooth loss: %.4f' % (ssim_loss_out,smooth_loss_out), Notify.ENDC)

                    ##save generated stuff##
                    if(total_step% (FLAGS.save_op_interval) ==0 and FLAGS.is_training):
                        postfix = "_" + str(total_step)

                        plt.imshow(dmap_out.squeeze())
                        plt.colorbar()
                        plt.savefig(self.op_dir + '/depth_map_out' + postfix +".png",dpi=500)
                        plt.close()

                        plt.imshow(dmap_gt.squeeze())
                        plt.colorbar()
                        plt.savefig(self.op_dir + '/depth_map_gt' + postfix +".png",dpi=500)
                        plt.close()

                        dmap_error = np.abs(dmap_gt.squeeze()  - dmap_out.squeeze())
                        gt_mask = np.where(dmap_gt.squeeze()==0)
                        dmap_error[gt_mask[0],gt_mask[1]] =0
                        plt.imshow(dmap_error)
                        plt.colorbar()
                        plt.savefig(self.op_dir + '/depth_map_error' + postfix +".png",dpi=500)
                        plt.close()

                        w1 = res_img[0, :, :, :].squeeze()
                        K = K_out=K_out.squeeze()

                        plt.imshow(np.mean(photo_error_map.squeeze(),axis=2))
                        plt.colorbar()
                        plt.savefig(self.op_dir + '/photo_map_out' + postfix +".png",dpi=500)
                        plt.close()

                        for view_cnt in range(warped_out.shape[0]):
                            w1 = warped_out[view_cnt, 0, :, :, :].squeeze()
                            w = (((w1 - w1.min()) * (1 / (w1.max() - w1.min()))) * 255.0).astype('uint8')
                            plt.imsave(self.op_dir + '/warped_out_' + str(view_cnt) + postfix + ".png", w)
                        for view_cnt in range(res_img.shape[0]):
                            w1 = res_img[view_cnt, :, :, :].squeeze()
                            w = (((w1 - w1.min()) * (1 / (w1.max() - w1.min()))) * 255.0).astype('uint8')
                            plt.imsave(self.op_dir + '/imgs_resized_' + str(view_cnt) + postfix + ".png", w)
                        for view_cnt in range(m_out.shape[0]):
                            m1 = m_out[view_cnt, :, :, :, :].squeeze()
                            plt.imsave(self.op_dir + '/mask_' + str(view_cnt) + postfix + ".png", m1)

                    # write summary
                    if step % (FLAGS.display * 10) == 0 and FLAGS.is_training:
                        summary_writer.add_summary(out_summary_op, total_step)

                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (self.training_sample_size - 1)) and FLAGS.is_training:
                        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                        self.saver.save(sess, checkpoint_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

def main(argv=None):
    """ program entrance """
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root,mode="training")
    val_sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root,mode="validation")

    # Shuffle
    random.seed(100)
    random.shuffle(sample_list)

    # Training entrance.
    dumvs_handle = DUMVS(training_list=sample_list,validation_list=val_sample_list)
    dumvs_handle.train_dumvs()


if __name__ == '__main__':

    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run(main)
