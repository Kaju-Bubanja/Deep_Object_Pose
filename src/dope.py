#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function
import yaml
import sys 

import numpy as np
import cv2

import rospy
import rospkg
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor_msg
from geometry_msgs.msg import PoseStamped

from PIL import Image
from PIL import ImageDraw

from shared_code import load_params, get_config_options

# Import DOPE code
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from cuboid import *
from detector import *

### Global Variables
g_bridge = CvBridge()
g_img = None
g_draw = None


### Basic functions
def __image_callback(msg):
    '''Image callback'''
    global g_img
    g_img = g_bridge.imgmsg_to_cv2(msg, "rgb8")
    # cv2.imwrite('img.png', cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0]-pointRadius, 
            point[1]-pointRadius, 
            point[0]+pointRadius, 
            point[1]+pointRadius
        ]
        g_draw.ellipse(xy, 
            fill=pointColor, 
            outline=pointColor
        )

def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across 
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)
    
    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)
    
    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius = 4)
    DrawDot(points[1], pointColor=color, pointRadius = 4)

    # draw x on the top 
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)


def run_dope_node(params, freq=5):
    '''Starts ROS node to listen to image topic, run DOPE, and publish DOPE results'''

    global g_img
    global g_draw

    pubs = {}
    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera, dist_coeffs, config_detect = get_config_options(params)

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] =\
            ModelData(
                model, 
                g_path2package + "/weights/" + params['weights'][model]
            )
        models[model].load_net_model()
        
        draw_colors[model] = \
            tuple(params["draw_colors"][model])
        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )
        pubs[model] = \
            rospy.Publisher(
                '{}/pose_{}'.format(params['topic_publishing'], model), 
                PoseStamped, 
                queue_size=10
            )
        pub_dimension[model] = \
            rospy.Publisher(
                '{}/dimension_{}'.format(params['topic_publishing'], model),
                String, 
                queue_size=10
            )

    # Start ROS publisher
    pub_rgb_dope_points = \
        rospy.Publisher(
            params['topic_publishing']+"/rgb_points", 
            ImageSensor_msg, 
            queue_size=10
        )
    
    # Starts ROS listener
    rospy.Subscriber(
        topic_cam, 
        ImageSensor_msg, 
        __image_callback
    )

    # Initialize ROS node
    rospy.init_node('dope_vis', anonymous=True)
    rate = rospy.Rate(freq)

    print ("Running DOPE...  (Listening to camera topic: '{}')".format(topic_cam)) 
    print ("Ctrl-C to stop")

    while not rospy.is_shutdown():
        if g_img is not None:
            # Copy and draw image
            img_copy = g_img.copy()
            im = Image.fromarray(img_copy)
            g_draw = ImageDraw.Draw(im)

            for m in models:
                # Detect object
                results = ObjectDetector.detect_object_in_image(
                            models[m].net, 
                            pnp_solvers[m],
                            g_img,
                            config_detect
                            )
                
                # Publish pose and overlay cube on image
                for i_r, result in enumerate(results):
                    if result["location"] is None:
                        continue
                    loc = result["location"]
                    ori = result["quaternion"]                    
                    msg = PoseStamped()
                    msg.header.frame_id = params["frame_id"]
                    msg.header.stamp = rospy.Time.now()
                    CONVERT_SCALE_CM_TO_METERS = 100
                    msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.orientation.x = ori[0]
                    msg.pose.orientation.y = ori[1]
                    msg.pose.orientation.z = ori[2]
                    msg.pose.orientation.w = ori[3]

                    # Publish
                    pubs[m].publish(msg)
                    pub_dimension[m].publish(str(params['dimensions'][m]))

                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        DrawCube(points2d, draw_colors[m])
                
            # Publish the image with results overlaid
            pub_rgb_dope_points.publish(
                CvBridge().cv2_to_imgmsg(
                    np.array(im)[...,::-1], 
                    "bgr8"
                )
            )
        rate.sleep()


if __name__ == "__main__":
    '''Main routine to run DOPE'''

    params = load_params()

    topic_cam = params['topic_camera']

    try :
        run_dope_node(params)
    except rospy.ROSInterruptException:
        pass
