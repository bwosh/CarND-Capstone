import cv2
import numpy as np
import os
import tensorflow as tf

from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image

from model import yolo_eval, tiny_yolo_body

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def letterbox_image(image, size):
    ih, iw = image.shape[0], image.shape[1]
    w, h = size
    scale = min(float(w)/float(iw), float(h)/float(ih))
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC) 
    new_image = np.ones((size[1],size[0],3), dtype='uint8')*128
    ox, oy = (w-nw)//2, (h-nh)//2
    new_image[oy:oy+nh,ox:ox+nw] = image
    return new_image

class TrafficLightsDetector():
    def __init__(self, score_threshold = 0.3, iou_threshold = 0.45):
        # Input parameters
        anchors_path = './data/tiny_yolo_anchors.txt'
        classes_path = './data/coco_classes.txt'
        model_path = './data/yolov3-tiny.h5'

        self.SELECTED_CLASS = 9

        # Start session

        config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.sess = K.get_session()
        self.K_learning_phase = K.learning_phase()

        # Preprare model
        anchors = get_anchors(anchors_path)
        classes = get_class(classes_path)
        self.model= tiny_yolo_body(Input(shape=(None,None,3)), len(anchors)//2, len(classes))
        self.model.load_weights(model_path) 

        # Prepare placeholders
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.ph_boxes, self.ph_scores, self.ph_classes = yolo_eval(self.model.output, anchors, len(classes), (416,416), 
                                        score_threshold=score_threshold, iou_threshold=iou_threshold)

    def detect_lights(self, cv2_image_bgr):
        # Image pre-processing
        boxed_image = letterbox_image(cv2_image_bgr, tuple(reversed((416,416))))
        image_data = (boxed_image[:,:,::-1]).astype('float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0) 

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.ph_boxes, self.ph_scores, self.ph_classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [416, 416],
                self.K_learning_phase: 0
            })

        # Preparing final result
        result = []
        for i in range(len(out_boxes)):
            if out_classes[i]==self.SELECTED_CLASS:
                top, left, bottom, right = out_boxes[i]
                label = out_classes[i]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(cv2_image_bgr.shape[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(cv2_image_bgr.shape[1], np.floor(right + 0.5).astype('int32'))             
                result.append(  ((left, top,right, bottom), out_scores[i])  )
        return boxed_image, result









    



