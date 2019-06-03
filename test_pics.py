import tensorflow as tf 
import numpy as np  
import argparse
import cv2
import random
import os
import utils.parameter_utils as parameter_utils
import utils.nms_utils as nms_utils
import utils.plot_utils as plot_utils
import utils.improve_utils as improve_utils
from yolov3_model import yolov3

ANCHOR_PATH = './data/anchors/location_anchors.txt'
CLASS_NAME = './data/class_name/cycle.names'
# RESTORE_PATH = './checkpoint/2019_4_18/useful/yolov3.ckpt-48600'
RESTORE_PATH = './checkpoint/test/yolov3_test.ckpt'

SAVE_PATH = './checkpoint/test/yolov3_multi_images.ckpt'

IMAGE_SHAPE = [416, 416]

input_image = '/home/ley/Documents/experiment_data/position_image_4_3/i931014.MRDC.png'
input_path = '/home/ley/Documents/experiment_data/position_image_4_3'

anchors = parameter_utils.parse_anchors(ANCHOR_PATH)
classes = parameter_utils.read_class_names(CLASS_NAME)
class_num = len(classes)
   
INPUT_NUM = 43
images = []
ori_images = []
result_path = []
image_size = []
num = 0
for image_path in os.listdir(input_path):
    if num < INPUT_NUM:
        full_path = os.path.join(input_path, image_path)
        ori_image = cv2.imread(full_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        ori_images.append(ori_image)
        image_size.append(list(ori_image.shape[:2]))

        image = tf.image.convert_image_dtype(ori_image, tf.float32)
        image = tf.image.resize(image, IMAGE_SHAPE)
        images.append(image)

        name = './demo_result/mutipictures/' + os.path.splitext(image_path)[0] + '_result.jpg'
        result_path.append(name)
        num += 1
    else:
        break
images = tf.convert_to_tensor(images)

input_data = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
# input_num = tf.placeholder(tf.int32)

yolo_model = yolov3(class_num, anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_data, False)
pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

pred_scores = pred_confs * pred_probs 

# boxes, scores, labels = nms_utils.nms(pred_boxes, pred_scores, class_num, max_boxes=30, score_thresh=0.5, iou_thresh=0.5)

pred_result = tf.concat([pred_boxes, pred_scores], axis=-1)
multi_result = tf.map_fn(lambda x: nms_utils.multi_nms(x, class_num, max_boxes=30, score_thresh=0.5, iou_thresh=0.5), pred_result) 
with tf.variable_scope('predict_output'):
    boxes = multi_result[:, :, :4]
    scores = tf.unstack(multi_result[:, :, 4:6], axis=-1)[0]
    labels = tf.cast(tf.unstack(multi_result[:, :, 4:6], axis=-1)[1], tf.int32)

# average model
ema = tf.train.ExponentialMovingAverage(decay=0.99)

# saver_to_restore = tf.train.Saver(ema.variables_to_restore())
saver_to_restore = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
    saver_to_restore.restore(sess, RESTORE_PATH)
    ######################################################
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./log/test', sess.graph)
    saver.save(sess, SAVE_PATH)
    ######################################################

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data:images.eval()})
    print([boxes_, scores_, labels_])

    for i, single_boxes in enumerate(boxes_):
        f = open("./compare.txt", 'a')
        f.writelines(result_path[i] + '\n')

        ori_image = ori_images[i]
        # rescale the coordinates to the original image
        single_boxes[:, 0] = single_boxes[:, 0] / float(IMAGE_SHAPE[1]) * float(image_size[i][1])
        single_boxes[:, 1] = single_boxes[:, 1] / float(IMAGE_SHAPE[0]) * float(image_size[i][0])
        single_boxes[:, 2] = single_boxes[:, 2] / float(IMAGE_SHAPE[1]) * float(image_size[i][1])
        single_boxes[:, 3] = single_boxes[:, 3] / float(IMAGE_SHAPE[0]) * float(image_size[i][0])

        center_points = []
        for j, box in enumerate(single_boxes):
            print('*' * 30)
            print(str(classes[labels_[i, j]]) + '  ' + str(box) + '  ' + str(scores_[i, j]))
            print('*' * 30)

            xmin, ymin, xmax, ymax = box
            x_center = int((xmin + xmax) / 2.)
            y_center = int((ymin + ymax) / 2.)

            # Correct the box to have a better result 
            if labels_[i, j] == 0 and x_center != 0 and y_center != 0:
                # plot_utils.ploy_rect(ori_image, box, color_table[labels_[i]])                  
                
                f.writelines(str(x_center) + " " + str(y_center) + '\n')
                
                center_points.append(improve_utils.corret_center(ori_image, x_center, y_center))

        f.close()

        # Remove the box   
        center_points = improve_utils.remove_extra_loc(ori_image, center_points)

        # Predict the lost box 
        center_points = improve_utils.predict_lost_box(ori_image, center_points)

        # Plot the center of boxes 
        plot_utils.plot_cross(ori_image, center_points)

        # plot_utils.ploy_rect(ori_image, box, color_table[labels_[i]])
            
        cv2.namedWindow('Detection_result', 0)
        # cv2.imshow('Detection_result', ori_image)
        cv2.imwrite(result_path[i], ori_image)
        # cv2.waitKey(0)