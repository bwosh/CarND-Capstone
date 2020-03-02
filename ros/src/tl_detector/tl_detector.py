#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import torch
import time

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.camera_image = None
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()
        rospy.spin()

    def image_cb(self, msg):
        self.has_image = True
        self.camera_image = msg
        state = self.is_red_light_visible()

    def preapre_tensor(self, cv_image):
        cv_image = cv2.resize(cv_image, (512,512))
        cv_image = (cv_image / 255.0)-0.5
        cv_image = cv_image.transpose(1,2,0)

        input = torch.tensor(cv_image, dtype=torch.float).unsqueeze(0)
        return input

    def detect(self, input):
        # TODO
        return []

    def classify(self, input, bbox):
        # TODO
        x1,y1,x2,y2 = bbox
        return []
        
    def is_red_light_visible(self):
        if(not self.has_image):
            return False

        time_a = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        original_shape = cv_image.shape
        cv_image = self.preapre_tensor(cv_image)
        detections = self.detect(cv_image)

        red_lights_colors = []
        for detection in detections:
            self.classify = self.classify(cv_image, detection)
        is_red = any(red_lights_colors)

        time_b = time.time()
        elapsed_ms = 1000.0 * (time_b-time_a)
        rospy.loginfo("Traffic Lights Execution time(ms):"+str(elapsed_ms))
        return is_red


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
