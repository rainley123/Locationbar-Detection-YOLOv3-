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

SAVE_PATH = './checkpoint/test/yolov3_test.ckpt'

IMAGE_SHAPE = [416, 416]

input_image = '/home/ley/Documents/experiment_data/position_image_4_3/i931014.MRDC.png'
# parser = argparse.ArgumentParser(description="YOLO-V3 test single picture")
# parser.add_argument("input_image", type=str, default= './demo_data/new_test/i722984.MRDC.png', help="The path of input image")

# args = parser.parse_args()

anchors = parameter_utils.parse_anchors(ANCHOR_PATH)
classes = parameter_utils.read_class_names(CLASS_NAME)
class_num = len(classes)
   
ori_image = cv2.imread(input_image)
ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
origin_img_size = ori_image.shape[:2]
image_name = os.path.splitext(os.path.basename(input_image))[0]
result_name = image_name + '_result.jpg'
result_path = './demo_result/c++_test/' + result_name

random.seed(0)
color_table = {}
for i in range(class_num):
    color_table[i] = [random.randint(0, 255) for RGB in range(3)]

image = tf.image.convert_image_dtype(ori_image, tf.float32)
image = tf.image.resize(image, IMAGE_SHAPE)
image = tf.expand_dims(image, axis=0)

input_data = tf.placeholder(tf.float32, shape=[1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
yolo_model = yolov3(class_num, anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_data, False)
pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

pred_scores = pred_confs * pred_probs 

boxes, scores, labels = nms_utils.nms(pred_boxes, pred_scores, class_num, max_boxes=30, score_thresh=0.5, iou_thresh=0.5)

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

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data:image.eval()})

    # rescale the coordinates to the original image
    boxes_[:, 0] = boxes_[:, 0] / float(IMAGE_SHAPE[1]) * float(origin_img_size[1])
    boxes_[:, 1] = boxes_[:, 1] / float(IMAGE_SHAPE[0]) * float(origin_img_size[0])
    boxes_[:, 2] = boxes_[:, 2] / float(IMAGE_SHAPE[1]) * float(origin_img_size[1])
    boxes_[:, 3] = boxes_[:, 3] / float(IMAGE_SHAPE[0]) * float(origin_img_size[0])

    center_points = []
    for i, box in enumerate(boxes_):
        print('*' * 30)
        print(str(classes[labels_[i]]) + '  ' + str(box) + '  ' + str(scores_[i]))
        print('*' * 30)

        xmin, ymin, xmax, ymax = box
        x_center = int((xmin + xmax) / 2.)
        y_center = int((ymin + ymax) / 2.)

        # Correct the box to have a better result 
        if labels_[i] == 0:
            # plot_utils.ploy_rect(ori_image, box, color_table[labels_[i]])
            center_points.append(improve_utils.corret_center(ori_image, x_center, y_center))

    # Remove the box   
    center_points = improve_utils.remove_extra_loc(ori_image, center_points)

    # Predict the lost box 
    center_points = improve_utils.predict_lost_box(ori_image, center_points)

    # Plot the center of boxes 
    plot_utils.plot_cross(ori_image, center_points)

    # plot_utils.ploy_rect(ori_image, box, color_table[labels_[i]])
        
    cv2.namedWindow('Detection_result', 0)
    # cv2.imshow('Detection_result', ori_image)
    cv2.imwrite(result_path, ori_image)
    # cv2.waitKey(0)