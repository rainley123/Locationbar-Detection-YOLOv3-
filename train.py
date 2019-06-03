import tensorflow as tf 
import numpy as np
from yolov3_model import yolov3
import utils.data_utils as data_utils
import utils.parameter_utils as parameter_utils
import utils.eval_utils as eval_utils
import utils.nms_utils as nms_utils
import matplotlib.pyplot as plt

# some paths
TRAIN_TFRECORDS = './tfrecords/train*'
VAL_TFRECORDS = './tfrecords/val.tfrecords'
ANCHOR_PATH = './data/anchors/location_anchors.txt'
CLASS_NAME = './data/class_name/cycle.names'
# RESTORE_PATH = './yolov3_weight/yolov3.ckpt'
RESTORE_PATH = './checkpoint/2019_4_17/useful/yolov3.ckpt-2376'
SAVE_DIR = './checkpoint/2019_4_18/yolov3.ckpt'

# some numbers
TRAIN_NUM = 173
VAL_NUM = 58

BATCH_SIZE = 10
IMAGE_SIZE = [416,416]
EPOCH = 2000
SHUFFLE_SIZE = 100
NUM_PARALLEL = 10

TRAIN_EVAL_INTERNAL = 300
VAL_EVAL_INTERNAL = 5
SAVE_INTERNAL = 5

# learning rate and optimizer
OPTIMIZER = 'adam'
LEARNING_RATE_INIT = 1e-4
LEARNING_RATE_TYPE = 'exponential'
LEARNING_RATE_DECAY_STEPS = 100
LEARNING_RATE_DECAY_RATE = 0.96
LEARNING_RATE_MIN = 1e-6

# variables part
RESTORE_PART = ['yolov3/darknet53_body']
UPDATE_PART = ['yolov3/yolov3_head']

with tf.Graph().as_default():
    # get the anchors and classes
    anchors = parameter_utils.parse_anchors(ANCHOR_PATH)
    classes = parameter_utils.read_class_names(CLASS_NAME)
    class_num = len(classes)

    # data pipline
    train_files = tf.train.match_filenames_once(TRAIN_TFRECORDS)
    train_dataset = tf.data.TFRecordDataset(train_files, buffer_size=4)
    train_dataset = train_dataset.map(lambda x : data_utils.parser(x, class_num, IMAGE_SIZE, anchors, 'train'), num_parallel_calls=NUM_PARALLEL)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE).repeat(4)

    val_files = [VAL_TFRECORDS]
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(lambda x : data_utils.parser(x, class_num, IMAGE_SIZE, anchors, 'val'), num_parallel_calls=NUM_PARALLEL)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    
    # create a public iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

    # get the element from the iterator
    image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    image.set_shape([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    # define the yolo_v3 model
    is_training = tf.placeholder(tf.bool)

    yolo_model = yolov3(class_num, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true)
    with tf.variable_scope('predict'):
        y_pred = yolo_model.predict(pred_feature_maps)

    ################
    # register the gpu nms operation here for the following evaluation scheme
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    with tf.variable_scope('nms'):
        gpu_nms_op = nms_utils.nms(pred_boxes_flag, pred_scores_flag, class_num)
    ################

    # create the global_step 
    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # define the learning rate and optimizer
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, decay_steps=LEARNING_RATE_DECAY_STEPS,
                                 decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True, name='exponential_learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # summary the loss and learning_rate
    tf.summary.scalar('total_loss', loss[0])
    tf.summary.scalar('loss_xy', loss[1])
    tf.summary.scalar('loss_wh', loss[2])
    tf.summary.scalar('loss_conf', loss[3])
    tf.summary.scalar('loss_class', loss[4])
    tf.summary.scalar('learning_rate', learning_rate)

    # restore and update var
    restore_vars = tf.contrib.framework.get_variables_to_restore(include=RESTORE_PART)
    update_vars = tf.contrib.framework.get_variables_to_restore(include=UPDATE_PART)
    # saver_to_restore = tf.train.Saver(var_list=restore_vars)
    saver_to_restore = tf.train.Saver()

    # average model 
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # update the BN vars
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = optimizer.minimize(loss[0], global_step=global_step, var_list=update_vars+restore_vars)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

    # set session config
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver_to_restore.restore(sess, RESTORE_PATH)
        saver = tf.train.Saver()

        write_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("./log/train", sess.graph)
        writer_val = tf.summary.FileWriter("./log/val", sess.graph)

        print('\n------------- start to train --------------\n')

        for epoch in range(EPOCH):
            sess.run(iterator.make_initializer(train_dataset))
            while True:
                try:
                    _, summary, y_pred_, y_true_, loss_, global_step_, learn_rate_= sess.run([train_op, write_op, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True})

                    writer_train.add_summary(summary, global_step=global_step_)
                    info = "Epoch: {}, global_step: {}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}, loss_l2: {:.3f}".format(
                    epoch, global_step_, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4], loss_[5])
                    print(info)

                    # evaluation on the train dataset
                    if (global_step_ + 1 ) % TRAIN_EVAL_INTERNAL == 0:
                        recall, precision = eval_utils.evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_, class_num, calc_now=True)
                        info = "===> batch recall: {:.3f}, batch precision: {:.3f} <===".format(recall, precision)
                        print(info)   
                except tf.errors.OutOfRangeError:
                        break

            if (epoch + 1) % SAVE_INTERNAL == 0:
                saver.save(sess, SAVE_DIR, global_step_)
            
            if (epoch + 1) % VAL_EVAL_INTERNAL == 0:
                sess.run(iterator.make_initializer(val_dataset))
                true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}
                val_loss = [0., 0., 0., 0., 0., 0.]
                while True:
                    try:
                        y_pred_, y_true_, loss_ = sess.run([y_pred, y_true, loss], feed_dict={is_training: False})
                        true_positive_dict_tmp, true_labels_dict_tmp, pred_labels_dict_tmp = eval_utils.evaluate_on_gpu(
                            sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_, class_num, calc_now=False)

                        true_positive_dict = parameter_utils.update_dict(true_positive_dict, true_positive_dict_tmp)
                        true_labels_dict = parameter_utils.update_dict(true_labels_dict, true_labels_dict_tmp)
                        pred_labels_dict = parameter_utils.update_dict(pred_labels_dict, pred_labels_dict_tmp)

                        val_loss = parameter_utils.list_add(val_loss, loss_)
                    except tf.errors.OutOfRangeError:
                        break
                    
                # make sure there is at least one ground truth object in each image
                # avoid divided by 0
                recall = float(sum(true_positive_dict.values())) / (sum(true_labels_dict.values()) + 1e-6)
                precision = float(sum(true_positive_dict.values())) / (sum(pred_labels_dict.values()) + 1e-6)


                info = "===> Epoch: {}, global_step: {}, recall: {:.3f}, precision: {:.3f}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}, loss_l2: {:.3f}".format(
                    epoch, global_step_, recall, precision, val_loss[0] / VAL_NUM, val_loss[1] / VAL_NUM, val_loss[2] / VAL_NUM, val_loss[3] / VAL_NUM, val_loss[4] / VAL_NUM, val_loss[5] / VAL_NUM)
                print(info)
                writer_val.add_summary(parameter_utils.make_summary('val_recall', recall), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('val_precision', precision), global_step=epoch)

                writer_val.add_summary(parameter_utils.make_summary('total_loss', val_loss[0] / VAL_NUM), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('loss_xy', val_loss[1] / VAL_NUM), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('loss_wh', val_loss[2] / VAL_NUM), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('loss_conf', val_loss[3] / VAL_NUM), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('loss_class', val_loss[4] / VAL_NUM), global_step=epoch)
                writer_val.add_summary(parameter_utils.make_summary('loss_l2', val_loss[5] / VAL_NUM), global_step=epoch)