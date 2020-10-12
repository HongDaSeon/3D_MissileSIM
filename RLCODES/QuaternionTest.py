#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# FDC Laboratory


import rospy                                          
import PSpincalc as spin
from pyquaternion import Quaternion
import numpy as np
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import pickle
import Missile_ENV_3D as MissileEnv
import AirCraft_ENV_3D as AirCrftEnv

from DaseonTypes import Vector3

from torch.distributions import Normal
from sensor_msgs.msg import Imu                                                     
from gazebo_msgs.msg import LinkState                                               
from collections import namedtuple

import matplotlib.pyplot as plt
import random as rd        
import vpython as vp       
import pdb         


def euler_to_quaternion(att):

        roll = att.x
        pitch = att.y
        yaw = att.z

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def RotationQuaternion(att, option="n_b"):
    _qx = Quaternion(axis=[1, 0, 0], angle=att.x)
    _qy = Quaternion(axis=[0, 1, 0], angle=att.y)
    _qz = Quaternion(axis=[0, 0, 1], angle=att.z)
    return _qz*_qy*_qx


