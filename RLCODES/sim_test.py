#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# FDC Laboratory
# Designed_by_Daseon_#

# rotating_cube.py

import rospy                                                                        # import ROS-python interface
import PSpincalc as spin
import numpy as np
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import pickle
from torch.distributions import Normal
from sensor_msgs.msg import Imu
 
from gazebo_msgs.msg import LinkState, LinkStates    
from geometry_msgs.msg import PoseStamped 
from collections import namedtuple
import matplotlib.pyplot as plt
import random as rd

Unity = True
Gazebo = False


att_x = 0.                                                                          # Initializing attitude
att_y = 0.
att_z = 0.
att_w = 0.

def processing_fnc(msg):                                                            # Subscriber Call back

    global att_x                                                                    # globalization of variables
    global att_y
    global att_z
    global att_w

    att_x = msg.orientation.x                                                       # put received message into the corresponding axis
    att_y = msg.orientation.y                                                       
    att_z = msg.orientation.z                                                       
    att_w = msg.orientation.w                                                       # Quarternion type angle


rospy.init_node('Animation') 


if Gazebo:
                                                           
    Dimension_dat = LinkStates()
    CraftDimension_dat = LinkState()                                                          # call the type class
    CraftDimension_dat.link_name = 'Aircraft::Aircraft::link_craft'                                             # ****link name should be identical of the one of at gazibo****

    CraftDimension_dat.pose.position.x = 0.0                                                  # Base position with catesian coordinate
    CraftDimension_dat.pose.position.y = 5.0
    CraftDimension_dat.pose.position.z = -5.0                                                  # levitate the model 5m so the model won't be dragged on the floor

    MissiDimension_dat = LinkState()
    MissiDimension_dat.link_name = 'MissileForTest::MissileForTest::link_missile'  

    MissiDimension_dat.pose.position.x = 0.0
    MissiDimension_dat.pose.position.y = -5.0
    MissiDimension_dat.pose.position.z = -5.0

    cmd__pub = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=1)  

    aquiz_dat = rospy.Subscriber('/mavros/imu/data', Imu, processing_fnc)     
    
if Unity:


    MissiDimension_dat = PoseStamped()

    MissiDimension_dat.pose.position.x = 0.0
    MissiDimension_dat.pose.position.y = 0.0
    MissiDimension_dat.pose.position.z = 0.0

    cmd__pub = rospy.Publisher('/MissPose', PoseStamped, queue_size=1)  

rate = rospy.Rate(100)                                                               # 30 loops/sec publish
k = 0
a = 0
while ~rospy.is_shutdown():                                                         # under rospy is running
    
    k = k+0.05

    att = spin.EA2Q([k, 0, 0])

    if Gazebo:
        CraftDimension_dat.pose.orientation.x = att[0][0]                                         # put the received data in a low at the buffer
        CraftDimension_dat.pose.orientation.y = att[0][1]
        CraftDimension_dat.pose.orientation.z = att[0][2]
        CraftDimension_dat.pose.orientation.w = att[0][3]

        MissiDimension_dat.pose.orientation.x = att[0][0]                                         # put the received data in a low at the buffer
        MissiDimension_dat.pose.orientation.y = att[0][1]
        MissiDimension_dat.pose.orientation.z = att[0][2]
        MissiDimension_dat.pose.orientation.w = att[0][3]

        Dimension_dat.name = [CraftDimension_dat.link_name, MissiDimension_dat.link_name]
        Dimension_dat.pose = [CraftDimension_dat.pose, MissiDimension_dat.pose]
        
        cmd__pub.publish(MissiDimension_dat)                                              # publish the message in the buffer

    if Unity:
        MissiDimension_dat = PoseStamped()

        MissiDimension_dat.pose.position.x = 0.0
        MissiDimension_dat.pose.position.y = 0.0
        MissiDimension_dat.pose.position.z = 0.0

        MissiDimension_dat.pose.orientation.x = att[0][0]
        MissiDimension_dat.pose.orientation.y = att[0][1]
        MissiDimension_dat.pose.orientation.z = att[0][2]
        MissiDimension_dat.pose.orientation.w = att[0][3]

        cmd__pub.publish(MissiDimension_dat)

    rate.sleep()                                                                    # corresponding delay to frequency

rospy.spin()  


def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]