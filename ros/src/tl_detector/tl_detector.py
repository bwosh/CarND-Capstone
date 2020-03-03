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

from transforms import decode_results
from dla import get_pose_net

class TLDetector(object):
    def __init__(self):
        rospy.loginfo("Initializing TLDetector")

        # ROS declarations
        rospy.init_node('tl_detector', )

        self.camera_image = None
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()

        # Read config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Detector model
        self.detector_max_frequency = self.config['detector_max_frequency']
        self.detection_threshold = self.config['detection_threshold']
        self.last_time = -1
        self.skipped_from_last = 0

        self.K = self.config['max_detections']
        self.device = self.config['device']
        device = torch.device(self.device)
        rospy.loginfo("Loading detecor model...")
        self.model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1).to(self.device)

        state_dict = torch.load("./data/detector.pth", map_location=device)
        self.model.load_state_dict(state_dict)
        self.model  = self.model.to(self.device)
        rospy.loginfo("Loaded detecor model.")

        rospy.spin()

    def image_cb(self, msg):
        self.has_image = True
        now = time.time()
        if 1/(now-self.last_time)<=self.detector_max_frequency:
            rospy.loginfo("TLDetector: Analysing new frame... Skipped frames:"+str(self.skipped_from_last))
            self.skipped_from_last=0
            self.camera_image = msg
            self.last_time = now
            state = self.is_red_light_visible()
        else:
            self.skipped_from_last+=1

    def preapre_tensor(self, cv_image):
        cv_image = cv2.resize(cv_image, (512,512))
        cv_image = (cv_image / 255.0)
        cv_image = cv_image.transpose(2,0,1)

        input = torch.tensor(cv_image, dtype=torch.float).unsqueeze(0)
        return input

    def detect(self, input):
        output = self.model.forward(input)[0]   
        output_hm = output['hm']
        output_wh = output['wh']

        # Decode results
        dets = decode_results(output_hm, output_wh, self.K)[0]
        return dets

    def classify(self, input, bbox):
        # TODO
        #x1,y1,x2,y2 = bbox
        result = False
        rospy.loginfo("CLS:"+str(bbox))
        rospy.loginfo("Is red light visible: "+str(result))
        return result
        
    def is_red_light_visible(self):
        if(not self.has_image):
            return False

        time_a = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        original_shape = cv_image.shape
        cv_image = self.preapre_tensor(cv_image)
        detections = self.detect(cv_image)

        red_lights_colors = [False]
        scores = detections['scores']
        bboxes = detections['bboxes']
        for di,_ in enumerate(detections):
            if scores[di] > self.detection_threshold:
                classification_result = self.classify(cv_image, bboxes[di])
                red_lights_colors.append(classification_result)
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
