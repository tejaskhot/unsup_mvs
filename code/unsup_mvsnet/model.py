import sys
import math
import tensorflow as tf
import numpy as np

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from homography_warping import get_homographies, homography_warping

FLAGS = tf.app.flags.FLAGS

def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """ get probability map from cost volume """

    def _repeat_(x, num_repeats):
        """ repeat each element num_repeats times """
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = tf.shape(cv)[1]

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    # d coordinate (floored and ceiled), batched & flattened
    d_coordinates = tf.reshape((depth_map - depth_start) / depth_interval, [-1])
    d_coordinates_left0 = tf.clip_by_value(tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates_left1 = tf.clip_by_value(d_coordinates_left0 - 1, 0, depth - 1)
    d_coordinates1_right0 = tf.clip_by_value(tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates1_right1 = tf.clip_by_value(d_coordinates1_right0 + 1, 0, depth - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_left1 = tf.stack(
        [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right1 = tf.stack(
        [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)

    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
    prob_map = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map

def inference(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=False)
        # ref_tower = UniNetDS2({'data': ref_image}, is_training=False, reuse=False)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
        # ref_tower = UniNetDS2({'data': ref_image}, is_training=False, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
        # view_tower = UniNetDS2({'data': view_image}, is_training=False, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), dim=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map,ref_tower,view_towers#, filtered_depth_map, probability_volume

def inference_3view(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """
    ##fix inference view count to 3 instead of FLAGS.view_num ###
    view_num =3

    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, 3):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, view_num- 1):
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / view_num
            ave_feature2 = ave_feature2 / view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), dim=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map,ref_tower,view_towers#, filtered_depth_map, probability_volume

def inference_mem(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
        # ref_tower = UniNetDS2({'data': ref_image}, is_training=False, reuse=False)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
        # ref_tower = UniNetDS2({'data': ref_image}, is_training=False, reuse=True)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
        # view_tower = UniNetDS2({'data': view_image}, is_training=False, reuse=True)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_features[view], homography)
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    filtered_depth_map = tf.cast(tf.greater_equal(prob_map, 0.8), dtype='float32') * estimated_depth_map

    return filtered_depth_map, prob_map

def depth_refine(init_depth_map, image, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.div(init_depth_map - depth_start_mat, depth_scale_mat)

    # resize normalized image to the same size of depth image
    resized_image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])

    # refinement network
    if is_master_gpu:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=tf.AUTO_REUSE)
    else:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=True)
    norm_depth_map = norm_depth_tower.get_output()

    # denormalize depth map
    refined_depth_map = tf.multiply(norm_depth_map, depth_scale_mat) + depth_start_mat

    return refined_depth_map

def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred))            # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum((masked_mae / interval) / denom)         # 1
    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_one_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

def mvsnet_loss(estimated_disp_image, disp_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(disp_image, estimated_disp_image, depth_interval)
    # less one accuracy
    less_one_accuracy = less_one_percentage(disp_image, estimated_disp_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(disp_image, estimated_disp_image, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy

def inverse_warping(img,left_cam, right_cam, depth):
    with tf.name_scope('inverse_warping'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        K_left = tf.squeeze(K_left,axis=1)


        K_left_inv = tf.matrix_inverse(K_left)
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        R_left =tf.squeeze(R_left, axis=1)
        t_left =tf.squeeze(t_left, axis=1)
        R_right = tf.squeeze(R_right, axis=1)
        t_right = tf.squeeze(t_right, axis=1)

        ## estimate egomotion by inverse composing R1,R2 and t1,t2
        R_rel = tf.matmul(R_right,R_left_trans)
        t_rel = tf.subtract(t_right, tf.matmul(R_rel,t_left))
        ## now convert R and t to transform mat, as in SFMlearner 
        batch_size = R_left.shape[0]
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([R_rel, t_rel], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)

        dims = tf.shape(img)
        batch_size, img_height, img_width = dims[0], dims[1], dims[2]
        depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
        grid = _meshgrid_abs(img_height, img_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
        cam_coords = _pixel2cam(depth, grid, K_left_inv)
        ones = tf.ones([batch_size, 1, img_height * img_width])
        cam_coords_hom = tf.concat([cam_coords, ones], axis=1)

        # Get projection matrix for target camera frame to source pixel frame
        hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])

        intrinsic_mat_hom = tf.concat(
            [K_left, tf.zeros([batch_size, 3, 1])], axis=2)
        intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
        proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, transform_mat)
        source_pixel_coords = _cam2pixel(cam_coords_hom,
                                         proj_target_cam_to_source_pixel)
        source_pixel_coords = tf.reshape(source_pixel_coords,
                                         [batch_size, 2, img_height, img_width])
        source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
        warped_right, mask = _spatial_transformer(img, source_pixel_coords)

    return warped_right,mask

def _cam2pixel(cam_coords, proj_c2p):
  """Transform coordinates in the camera frame to the pixel frame."""
  pcoords = tf.matmul(proj_c2p, cam_coords)
  x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
  y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
  z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
  x_norm = x / (z + 1e-10)
  y_norm = y / (z + 1e-10)
  pixel_coords = tf.concat([x_norm, y_norm], axis=1)
  return pixel_coords

def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
  """Transform coordinates in the pixel frame to the camera frame."""
  cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth
  return cam_coords

def _meshgrid_abs(height, width):
  """Meshgrid in the absolute coordinates."""
  x_t = tf.matmul(
      tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(
      tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  x_t_flat = tf.reshape(x_t, (1, -1))
  y_t_flat = tf.reshape(y_t, (1, -1))
  ones = tf.ones_like(x_t_flat)
  grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
  return grid

def _spatial_transformer(img, coords):
  """A wrapper over binlinear_sampler(), taking absolute coords as input."""
  img_height = tf.cast(tf.shape(img)[1], tf.float32)
  img_width = tf.cast(tf.shape(img)[2], tf.float32)
  px = coords[:, :, :, :1]
  py = coords[:, :, :, 1:]
  # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
  px = px / (img_width - 1) * 2.0 - 1.0
  py = py / (img_height - 1) * 2.0 - 1.0
  output_img, mask = _bilinear_sampler(img, px, py)
  return output_img, mask

def _bilinear_sampler(im, x, y, name='blinear_sampler'):
  """Perform bilinear sampling on im given list of x, y coordinates.
  Implements the differentiable sampling mechanism with bilinear kernel
  in https://arxiv.org/abs/1506.02025.
  x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
  For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
  and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
  Args:
    im: Batch of images with shape [B, h, w, channels].
    x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
    y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
    name: Name scope for ops.
  Returns:
    Sampled image with shape [B, h, w, channels].
    Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
      in the mask indicates that the corresponding coordinate in the sampled
      image is valid.
  """
  with tf.variable_scope(name):
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # Constants.
    batch_size = tf.shape(im)[0]
    _, height, width, channels = im.get_shape().as_list()

    x = tf.to_float(x)
    y = tf.to_float(y)
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    mask = tf.logical_and(
        tf.logical_and(x0 >= zero, x1 <= max_x),
        tf.logical_and(y0 >= zero, y1 <= max_y))
    mask = tf.to_float(mask)

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = tf.range(batch_size) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    # And finally calculate interpolated values.
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
    mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))

    return output, mask

def ssim(x, y, mask):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = tf.layers.average_pooling2d(x, 3, 1, 'VALID')
    mu_y = tf.layers.average_pooling2d(y, 3, 1, 'VALID')
    sigma_x = tf.layers.average_pooling2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = tf.layers.average_pooling2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = tf.layers.average_pooling2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    ssim_mask = tf.layers.average_pooling2d(mask, 3, 1, 'VALID')
    return ssim_mask * tf.clip_by_value((1 - ssim) / 2, 0, 1)

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy

def depth_smoothness(depth, img,lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-(lambda_wt * tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True)))
    weights_y = tf.exp(- (lambda_wt*tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

def smooth_loss_sfmlearner(pred_disp):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    return tf.reduce_mean(tf.abs(dx2)) + \
           tf.reduce_mean(tf.abs(dxdy)) + \
           tf.reduce_mean(tf.abs(dydx)) + \
           tf.reduce_mean(tf.abs(dy2))

def compute_reconstr_loss(warped,ref,mask,simple=1):


    if(simple):
        return tf.losses.huber_loss((warped * mask),(ref*mask),delta=0.6,reduction=tf.losses.Reduction.MEAN)
    else:
        alpha =0.5
        ref_dx,ref_dy = gradient(ref*mask)
        warped_dx,warped_dy = gradient(warped*mask)
        photo_loss = tf.losses.huber_loss((warped * mask),(ref*mask),delta=0.6,reduction=tf.losses.Reduction.MEAN)
        grad_loss = tf.losses.huber_loss(warped_dx, ref_dx, delta=0.2, reduction=tf.losses.Reduction.MEAN) + \
                    tf.losses.huber_loss(warped_dy, ref_dy, delta = 0.2, reduction = tf.losses.Reduction.MEAN)

        return ((1-alpha)*photo_loss + (alpha)*grad_loss)

def compute_reconstr_loss_map(warped,ref,mask,simple = 0):

    if (simple):
        return tf.losses.absolute_difference((warped * mask), (ref * mask), reduction=tf.losses.Reduction.MEAN)

    else:

        alpha =0.5
        ref_dy,ref_dx = tf.image.image_gradients(ref*mask)
        warped_dy,warped_dx = tf.image.image_gradients(warped*mask)
        photo_loss = tf.losses.huber_loss((warped * mask),(ref*mask),delta=0.6,reduction=tf.losses.Reduction.NONE)

        grad_loss_x = tf.abs(warped_dx - ref_dx)
        grad_loss_y= tf.abs(warped_dy - ref_dy)
        grad_loss = grad_loss_x + grad_loss_y
        return ((1-alpha)*photo_loss + (alpha)*grad_loss)


