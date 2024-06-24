#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' 
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.2.1
Date    : Jan 20, 2019

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate

    The point correspondences will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/img_corners.npy
    - PKG_PATH/calibration_data/lidar_camera_calibration/pcl_corners.npy

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.npz
    --> 'euler' : euler angles (3, )
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ roslaunch lidar_camera_calibration display_camera_lidar_calibration.launch

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import tf

# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import tf2_ros
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Path
from sklearn.mixture import GaussianMixture
from geometry_msgs.msg import PoseStamped

# Global variables
OUSTER_LIDAR = True
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
CAMERA_MODEL = image_geometry.PinholeCameraModel()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

gm = GaussianMixture(n_components=2,    # default cluster : 1
                        max_iter=10,
                        init_params='kmeans',     # max_iter 10
                    )

net = cv2.dnn_DetectionModel("/home/krm/detectppl/yolov4-tiny.cfg", "/home/krm/detectppl/yolov4-tiny.weights")
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

with open('/home/krm/cal_ws/src/lidar_camera_calibration/scripts/coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')
    
INT = 0

def mkmat(rows, cols, L):
    mat = np.matrix(L, dtype='float64')
    mat.resize((rows,cols))
    return mat

def project3dToPixel(point, intrinsic):
    src = mkmat(4, 1, [point[0], point[1], point[2], 0.1])
    dst = intrinsic * src
    # dst = src
    x = dst[0,0]
    y = dst[1,0]
    w = dst[2,0]
    if w != 0:
        return (x / w, y / w)
    else:
        return (float('nan'), float('nan'))
 

'''
Projects the point cloud on to the image plane using the extrinsics

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs:
    Projected points published on /sensors/camera/camera_lidar topic
'''

def project_point_cloud_short(velodyne, img_msg, image_pub, net, names, end_pub):
    br = tf.TransformBroadcaster()

    global INT
    # Read image using CV bridge
    try:
        original_img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
        img = original_img.copy()
    except CvBridgeError as e: 
        rospy.logerr(e)
        return
    
    try:
        transform = TF_BUFFER.lookup_transform('world', 'velodyne', rospy.Time())
        velodyne = do_transform_cloud(velodyne, transform)
    except tf2_ros.LookupException:
        pass

    # Extract points from message
    points3D = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points3D = np.asarray(points3D.tolist())
    
    # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
    if OUSTER_LIDAR: points3D = points3D.reshape(-1, 9)[:, :4]

    # Filter points in front of camera
    inrange = np.where((points3D[:, 2] > 0) &
                       (points3D[:, 2] < 6) &
                       (np.abs(points3D[:, 0]) < 6) &
                       (np.abs(points3D[:, 1]) < 6))
    max_intensity = np.max(points3D[:, 3])
    points3D = points3D[inrange[0]]

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, 3] / max_intensity) * 255

    # Project to 2D and filter points within image boundaries
    P = np.matrix([265.875517, 0.0, 464.938881, 0.0, 0.0, 315.544707, 222.051443, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)       # obtained from calibration process
    points2D = [project3dToPixel(point, P) for point in points3D[:, :3] ]

    # Draw box on the image
    img = original_img.copy()
    person_indices = []
    classes, confidences, boxes = net.detect(img, confThreshold=0.1, nmsThreshold=0.4)
    max_height = 0
    for idx, (classId, confidence, box) in enumerate(zip(classes, confidences, boxes)):
        if classId == 0 and confidence > 0.85:
            label = '%.2f' % confidence
            label = '%s: %s' % (names[classId], label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            top = max(top, labelSize[1])
            person_indices.append([idx, box, top])
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.rectangle(img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            bbox_img = img.copy()


    # Project to 2D and filter points within image boundaries
    P = np.matrix([265.875517, 0.0, 464.938881, 0.0, 0.0, 315.544707, 222.051443, 0.0, 0.0, 0.0, 1.0, 0.0], dtype='float64').reshape(3, 4)
    points2D = [project3dToPixel(point, P) for point in points3D[:, :3] ]
    points2D = np.asarray(points2D)

    new_points2D = []
    optimal_3D = []
    max_height = 0
    for idx, person in enumerate(person_indices):
        idx, box, top = person
        left, top, width, height = box
        inrange = np.where((points2D[:, 0] >= left) &
                        (points2D[:, 1] >= top) &
                        (points2D[:, 0] < left+width) &
                        (points2D[:, 1] < top+height))
        new_points3D = points3D[inrange[0]].round().astype('int')   # in box 3D points
        if max_height < height:
            max_height = height
            new_points2D = points2D[inrange[0]].round().astype('int')
            optimal_3D = new_points3D

    
    if (optimal_3D != [] and inrange[0].shape[0] > 1):
        indices = gm.fit_predict(optimal_3D[:, 2].reshape(-1,1))
        if (gm.means_.shape[0] > 1) : 
            idx = 0
            if np.sort(gm.means_.reshape(-1))[0] != gm.means_.reshape(-1)[0]:
                idx = 1
            mean = gm.means_.reshape(-1)[idx]    # smallest mean
            output_indices = np.where(indices == idx)
            print(f'gaussian cluster numbers is {gm.means_.shape[0]}')
        else: 
            mean = optimal_3D[:, :2].mean() 

        img = bbox_img.copy()
        for i in range(len(new_points2D[output_indices[0]])):
            cv2.circle(img, tuple(new_points2D[output_indices[0]][i]), 2, tuple(colors[i]), -1)


        try:
            image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e: 
            rospy.logerr(e)

        data = PoseStamped()
        nav_data = Path()
        data.header.stamp = rospy.get_rostime()
        nav_data.header.stamp = rospy.get_rostime()
        data.pose.position.x = np.median(optimal_3D[:, 2])
        data.pose.position.y = -np.mean(optimal_3D[:, 0])
        data.pose.position.z = -np.mean(optimal_3D[:, 1])
        nav_data.poses.append(data)
        end_pub.publish(nav_data)
        
        br.sendTransform((mean, -np.mean(optimal_3D[:, 0]), -np.mean(optimal_3D[:, 1])),
                        (0.0, 0.0, 0.0, 1.0),
                        rospy.Time.now(),
                        "target",
                        "base_link")
    INT += 1



'''
Callback function to publish project image and run calibration

Inputs:
    image - [sensor_msgs/Image] - ROS sensor image message
    camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs: None
'''
def callback(image, velodyne, image_pub=None, end_pub = None):
    global CAMERA_MODEL, FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER

    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False

        # Setup camera model
        rospy.loginfo('Setting up camera model')

        # TF listener
        rospy.loginfo('Setting up static transform listener')
        TF_BUFFER = tf2_ros.Buffer()
        TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)

    # YOLO detection
    project_point_cloud_short(velodyne, image, image_pub, net, names, end_pub)

'''
The main ROS node which handles the topics

Inputs:
    camera_info - [str] - ROS sensor camera info topic
    image_color - [str] - ROS sensor image topic
    velodyne - [str] - ROS velodyne PCL2 topic
    camera_lidar - [str] - ROS projected points image topic

Outputs: None
'''
def listener(image_color, velodyne_points, camera_lidar=None):
    # Start node
    rospy.init_node('calibrate_camera_lidar', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    rospy.loginfo('Projection mode: %s' % PROJECT_MODE)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('PointCloud2 topic: %s' % velodyne_points)
    rospy.loginfo('Output topic: %s' % camera_lidar)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    # Publish output topic
    image_pub = None
    if camera_lidar: 
        image_pub = rospy.Publisher(camera_lidar, Image, queue_size=5)
        end_pub = rospy.Publisher("/scout/local_goal", Path, queue_size=1)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, velodyne_sub], queue_size=5, slop=100000000)
    ats.registerCallback(callback, image_pub, end_pub)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':

    camera_info = rospy.get_param('camera_info_topic')
    image_color = rospy.get_param('image_color_topic')
    velodyne_points = rospy.get_param('velodyne_points_topic')
    camera_lidar = rospy.get_param('camera_lidar_topic')
    PROJECT_MODE = bool(rospy.get_param('project_mode'))

    # Start subscriber
    listener(image_color, velodyne_points, camera_lidar)
