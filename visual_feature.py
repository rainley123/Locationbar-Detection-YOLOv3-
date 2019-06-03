import tensorflow as tf 
import numpy as np 
from yolov3_model import yolov3
import utils.parameter_utils as parameter_utils
import matplotlib.pyplot as plt
import cv2

input_image = '/home/ley/Documents/experiment_data/paper_need/i593896.MRDC.png'

ANCHOR_PATH = './data/anchors/location_anchors.txt'
CLASS_NAME = './data/class_name/cycle.names'
RESTORE_PATH = './checkpoint/2019_4_18/useful/yolov3.ckpt-48600'

IMAGE_SHAPE = [416, 416]

anchors = parameter_utils.parse_anchors(ANCHOR_PATH)
classes = parameter_utils.read_class_names(CLASS_NAME)
class_num = len(classes)
   
ori_image = cv2.imread(input_image)
ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
origin_img_size = ori_image.shape[:2]

image = tf.image.convert_image_dtype(ori_image, tf.float32)
image = tf.image.resize(image, IMAGE_SHAPE)
image = tf.expand_dims(image, axis=0)

def upsample(inputs, out_shape):
    new_height, new_width = out_shape[0], out_shape[1]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs

input_data = tf.placeholder(tf.float32, shape=[1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
yolo_model = yolov3(class_num, anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_data, False)
    feature_map1, feature_map2, feature_map3 = pred_feature_maps
    # feature_map1 = upsample(feature_map1, IMAGE_SHAPE)
    # feature_map2 = upsample(feature_map2, IMAGE_SHAPE)
    # feature_map3 = upsample(feature_map3, IMAGE_SHAPE)
    
# average model 
ema = tf.train.ExponentialMovingAverage(decay=0.99)

saver_to_restore = tf.train.Saver(ema.variables_to_restore())

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
    saver_to_restore.restore(sess, RESTORE_PATH)

    map1, map2, map3 = sess.run([feature_map1, feature_map2, feature_map3], feed_dict={input_data:image.eval()})

    map1 = np.squeeze(map1)   
    map2 = np.squeeze(map2)
    map3 = np.squeeze(map3)

    map1 = np.sum(map1, axis=-1)
    map2 = np.sum(map2, axis=-1)
    map3 = np.sum(map3, axis=-1)
    
    plt.figure('feature')
    plt.imshow(map3)
    plt.savefig('/home/ley/PycharmProjects/Location_Dection/paper_figure/map3.jpg')
    # plt.subplot(3, 1, 1)
    # plt.imshow(map1)

    # plt.subplot(3, 1, 2)
    # plt.imshow(map2)

    # plt.subplot(3, 1, 3)
    # plt.imshow(map3)

    plt.show()