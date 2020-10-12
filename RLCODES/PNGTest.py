#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#  _______ _____    ______    _            _                                            
# (_______|____ \  / _____)  | |          | |                     _                     
#  _____   _   \ \| /        | |      ____| | _   ___   ____ ____| |_  ___   ____ _   _ 
# |  ___) | |   | | |        | |     / _  | || \ / _ \ / ___) _  |  _)/ _ \ / ___) | | |
# | |     | |__/ /| \_____   | |____( ( | | |_) ) |_| | |  ( ( | | |_| |_| | |   | |_| |
# |_|     |_____/  \______)  |_______)_||_|____/ \___/|_|   \_||_|\___)___/|_|    \__  |
#                                                                                (____/ 
# FDC Laboratory
#    _____ ____     __  ____           _ __        ____  __ 
#   |__  // __ \   /  |/  (_)_________(_) /__     / __ \/ / 
#    /_ </ / / /  / /|_/ / / ___/ ___/ / / _ \   / /_/ / /  
#  ___/ / /_/ /  / /  / / (__  |__  ) / /  __/  / _, _/ /___
# /____/_____/  /_/  /_/_/____/____/_/_/\___/  /_/ |_/_____/
                                                          
# Designed_by_Daseon_#


import sys
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

from DaseonTypes import Vector3, ASCIIart

from torch.distributions import Normal
from sensor_msgs.msg import Imu                                                     
from gazebo_msgs.msg import LinkState   
from geometry_msgs.msg import PoseStamped              
from misssim.msg import LearningMonitor                              
from collections import namedtuple

import matplotlib.pyplot as plt
import random as rd        
import vpython as vp       
import pdb                             

# Argument List
#   1. GPU number
#   2. DataStream Discription

# basic initialization+++++++++++++++++++++++++++++++++++++++++++++
ASCIIart.FDCLAB()
# Parameters +++++++++++++++#
realtimeSim = False          #
rospublish = True           #
gazebosim = False           #
Unitysim = True             #
vpythonsim = False          #
# ++++++++++++++++++++++++++#

SessionIdentificationNumber = int(rd.random()*1000000000)

TransmitSwitcher = 0
dtSim = 0.05
animcounter = 0
aclr = 1e-7
crlr = 1e-6
# nn param
gpu_num = int(sys.argv[1])
mu_now = 0
hitCount = 0
saveCount = 0
solved = False
slept = False

tau = 0.05
max_step_count = 120

# Function inits ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def processing_fnc(msg):        #                                                     
                                #
    global att_x                #                                                    
    global att_y                #
    global att_z                #
    global att_w                #
                                #
    att_x = msg.orientation.x   #                                                    
    att_y = msg.orientation.y   #                                                    
    att_z = msg.orientation.z   #                                                    
    att_w = msg.orientation.w   #

def PPNG(N,Vr,Vm,Om,Qnb):
    Acc = ( - N * Vr.mag * (np.cross(Qnb.inverse.rotate(Vm.direction.vec), Om.vec)) )
    return Acc
    
def init_picker():
    minradius = 10000
    maxradius = 20000
    minSpeed  = 200
    maxSpeed  = 600
    
    rand_R   = Vector3(rd.random()*(maxradius-minradius) + minradius,0.,0.)
    
    rand_lam = Vector3(0., (rd.random()-0.5)*m.pi, rd.random()*2*m.pi)
    
    init_pos = Vector3.cast(RotationQuaternion(rand_lam).rotate(rand_R.vec))

    head_vec = Vector3.cast(init_pos.vec*-1)
    gazeazim, gazeelev = azimNelev(head_vec)
    
    head_ang = Vector3(0., gazeelev, gazeazim)
    Qhead = RotationQuaternion(head_ang)

    hed_seed = Vector3(0., (rd.random()-0.5)*m.pi, (rd.random()-0.5)*m.pi)
    Qseed = RotationQuaternion(hed_seed)
    Qrotation = Qhead*Qseed
    rand_hed = Qrotation.rotate(rand_R.vec)

    headazim, headelev = azimNelev(Vector3.cast(rand_hed))

    init_hed = Vector3(0., headelev, headazim)
    
    #init_hed = Vector3(0., rand_lam.y-m.pi+(rd.random()-0.5)*m.pi, rand_lam.z-m.pi+(rd.random()-0.5)*m.pi)


    
    init_spd = (maxSpeed - minSpeed)*rd.random() + minSpeed

    return init_pos, init_hed, init_spd  

def Transmit2Unity(craft_dat,missile_dat):
    global TransmitSwitcher
    if Unitysim & ~rospy.is_shutdown():
        
        craftAtt    = euler_to_quaternion(craft_dat.att)
        missileAtt  = euler_to_quaternion(missile_dat.att)

        Craft_Dimension_dat.pose.position.x = craft_dat.pos.x
        Craft_Dimension_dat.pose.position.y = craft_dat.pos.y
        Craft_Dimension_dat.pose.position.z = craft_dat.pos.z

        Craft_Dimension_dat.pose.orientation.x = craftAtt[0]                                         
        Craft_Dimension_dat.pose.orientation.y = craftAtt[1]
        Craft_Dimension_dat.pose.orientation.z = craftAtt[2]
        Craft_Dimension_dat.pose.orientation.w = craftAtt[3]

        Missi_Dimension_dat.pose.position.x = missile_dat.pos.x
        Missi_Dimension_dat.pose.position.y = missile_dat.pos.y
        Missi_Dimension_dat.pose.position.z = missile_dat.pos.z

        Missi_Dimension_dat.pose.orientation.x = missileAtt[0]                                         
        Missi_Dimension_dat.pose.orientation.y = missileAtt[1]
        Missi_Dimension_dat.pose.orientation.z = missileAtt[2]
        Missi_Dimension_dat.pose.orientation.w = missileAtt[3]

        cmd_missi_pub.publish(Missi_Dimension_dat)
        cmd_craft_pub.publish(Craft_Dimension_dat) 
        '''
        if TransmitSwitcher == 0:
            cmd_missi_pub.publish(Missi_Dimension_dat)
            TransmitSwitcher = 1
        else:
            cmd_craft_pub.publish(Craft_Dimension_dat)                                              
            TransmitSwitcher = 0
        '''
    if realtimeSim: rate.sleep() 

def Transmit2Gazebo(craft_dat,missile_dat):
    global TransmitSwitcher
    if gazebosim & ~rospy.is_shutdown():
        #pdb.set_trace()
        craftAtt    = euler_to_quaternion(craft_dat.att)
        missileAtt  = euler_to_quaternion(missile_dat.att)
        
        Craft_Dimension_dat.pose.position.x = craft_dat.pos.x
        Craft_Dimension_dat.pose.position.y = craft_dat.pos.y
        Craft_Dimension_dat.pose.position.z = craft_dat.pos.z

        Craft_Dimension_dat.pose.orientation.x = craftAtt[0]                                         
        Craft_Dimension_dat.pose.orientation.y = craftAtt[1]
        Craft_Dimension_dat.pose.orientation.z = craftAtt[2]
        Craft_Dimension_dat.pose.orientation.w = craftAtt[3]

        Missi_Dimension_dat.pose.position.x = missile_dat.pos.x
        Missi_Dimension_dat.pose.position.y = missile_dat.pos.y
        Missi_Dimension_dat.pose.position.z = missile_dat.pos.z

        Missi_Dimension_dat.pose.orientation.x = missileAtt[0]                                         
        Missi_Dimension_dat.pose.orientation.y = missileAtt[1]
        Missi_Dimension_dat.pose.orientation.z = missileAtt[2]
        Missi_Dimension_dat.pose.orientation.w = missileAtt[3]
        
        if TransmitSwitcher == 0:
            cmd_missi_pub.publish(Missi_Dimension_dat)
            TransmitSwitcher = 1
        else:
            cmd_craft_pub.publish(Craft_Dimension_dat)                                              
            TransmitSwitcher = 0

    if realtimeSim: rate.sleep()             

#++++++++++++++Function init finished++++++++++++++++++++++++++++++++++++++++

def azimNelev(vec):
        azim = m.atan2( vec.y, vec.x)
        elev = m.atan2( -vec.z, m.sqrt( vec.x**2 + vec.y**2))
        return azim, elev

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

def StreamDataFunnel(Subject, ep, t, Model, Trwd, Mrwd, done):
    
    Learning_dat_basic.episode = int(ep)
    Learning_dat_basic.time = t
    
    Learning_dat_basic.dimension.pose.x = Model.pos.x
    Learning_dat_basic.dimension.pose.y = Model.pos.y
    Learning_dat_basic.dimension.pose.z = Model.pos.z
    quatAtt = euler_to_quaternion(Model.att)
    Learning_dat_basic.dimension.att.x = quatAtt[0]
    Learning_dat_basic.dimension.att.y = quatAtt[1]
    Learning_dat_basic.dimension.att.z = quatAtt[2]
    Learning_dat_basic.dimension.att.w = quatAtt[3]
    Learning_dat_basic.dimension.acc.x = Model.acc.x
    Learning_dat_basic.dimension.acc.y = Model.acc.y
    Learning_dat_basic.dimension.acc.z = Model.acc.z
    Learning_dat_basic.TDreward = Trwd
    Learning_dat_basic.MCreward = Mrwd
    Learning_dat_basic.done = done
    
    if Subject == "Missile":
        MissiStreamPub.publish(Learning_dat_basic)
    elif Subject == "Fighter":
        CraftStreamPub.publish(Learning_dat_basic)


#=======================================================================
# ROS Initialize 
rospy.init_node('SimVisualization')                                                        
rate = rospy.Rate(1000)
#-----------------------------------------------------------------------
if gazebosim:
    Craft_Dimension_dat = LinkState()                                                          
    Craft_Dimension_dat.link_name = 'Aircraft::Aircraft::link_craft'                      
    Missi_Dimension_dat = LinkState()
    Missi_Dimension_dat.link_name = 'MissileForTest::MissileForTest::link_missile'                         
if Unitysim:
    Craft_Dimension_dat = PoseStamped()
    Missi_Dimension_dat = PoseStamped()
#-----------------------------------------------------------------------
Craft_Dimension_dat.pose.position.x = 0.0                                                  
Craft_Dimension_dat.pose.position.y = 0.
Craft_Dimension_dat.pose.position.z = 0.                                                  

Missi_Dimension_dat.pose.position.x = 0.0
Missi_Dimension_dat.pose.position.y = -5.0
Missi_Dimension_dat.pose.position.z = -5.0
#-----------------------------------------------------------------------
if gazebosim:
    cmd_craft_pub = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=1)
    cmd_missi_pub = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=1)
if Unitysim:
    cmd_craft_pub = rospy.Publisher('/CraftPose', PoseStamped, queue_size=1)
    cmd_missi_pub = rospy.Publisher('/MissPose', PoseStamped, queue_size=1)
#=======================================================================

#=======================================================================
# Data Stream Initializer
Learning_dat_basic = LearningMonitor()
Craft_Learning_dat = LearningMonitor()
Missi_Learning_dat = LearningMonitor()
#-----------------------------------------------------------------------
Learning_dat_basic.SessionDiscription = sys.argv[2]
Learning_dat_basic.SessionIdentifier = SessionIdentificationNumber

Learning_dat_basic.deltatime = dtSim
Learning_dat_basic.GPU = gpu_num

Learning_dat_basic.done = False
#-----------------------------------------------------------------------
CraftStreamPub = rospy.Publisher('/LearningInfo/Craft', LearningMonitor, queue_size=1)
MissiStreamPub = rospy.Publisher('/LearningInfo/Missile', LearningMonitor, queue_size=1)

#=======================================================================

aquiz_dat = rospy.Subscriber('/mavros/imu/data', Imu, processing_fnc)    # Garbage

# Instances +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MissileModel_1 = MissileEnv.Missile_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(MissileModel_1)
FighterModel_1 = AirCrftEnv.Craft_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(FighterModel_1)
MissileSeeker_1 = MissileEnv.Seeker(MissileModel_1, FighterModel_1)
FighterSeeker_1 = AirCrftEnv.Seeker(FighterModel_1, MissileModel_1)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i_ep in range(10000):
    
    t = 0 ##sec
    score = 0
    posM, hedM, spdM = init_picker()
    print('===========================================================================')
    print('EP : ', i_ep, '| Init : ','\n\t\t',posM,'\n\t\t',hedM,'\n\t\t',spdM)
    
    MissileSeeker_1.impactR = 50000
    
    closestR = 999999
    step_count = 0
    energy = 0
    
    print('-------------------------------PNG-----------------------------------------')

    MissileModel_1.reset(posM, hedM, spdM, reset_flag = True)
    t = 0 ##sec
    step_count = 0
    MissileSeeker_1.impactR = 50000
    _, Look, LOSdot, _, state = MissileSeeker_1.seek()

    energy_term = []
    energy = 0
    pngtime = []
    pngacc  = []
    print(MissileSeeker_1.Look.z, MissileSeeker_1.Look.y)

    while t<max_step_count:
        
        Acc_cmd = Vector3.cast(np.clip(PPNG(3, \
                                    MissileSeeker_1.Vvec, \
                                    MissileSeeker_1.direcVec, \
                                    MissileSeeker_1.dLOS, \
                                    MissileModel_1.Qnb), \
                                    -99,    99))
        
        #pdb.set_trace()
        MissileModel_1.simulate(Acc_cmd)
        
        Rmag, Look, LOSdiot, _, state_ = MissileSeeker_1.seek()
        if gazebosim: Transmit2Gazebo(FighterModel_1, MissileModel_1)
        if Unitysim: Transmit2Unity(FighterModel_1, MissileModel_1)
        
        #print(str(i_ep)+'  ,'+'time :' +format(t,'.2f')+'  ,'+str(MissileModel_1.pos)+'\n'+str(Acc_cmd))
        
        Mt_reward, Mm_reward, M_done, M_is_hit = MissileSeeker_1.spit_reward(Acc_cmd)
        Ct_reward, Cm_reward, C_done, C_is_hit = FighterSeeker_1.spit_reward(Vector3(0,0,0))
        
        t = t + dtSim
        StreamDataFunnel("Missile", i_ep, t, MissileModel_1, Mt_reward, Mm_reward,M_done)
        StreamDataFunnel("Fighter", i_ep, t, FighterModel_1, Ct_reward, Cm_reward,C_done)
        
        
        if M_done:
            print(M_is_hit)
            break

        
    
    print('===========================================================================\n\n')

quit()
rospy.spin()  

#++++++++++++++++++++++++++++++Functions++++++++++++++++++++++++++++++

