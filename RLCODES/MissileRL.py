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

# RL for a Missile

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

FilePath = "/root/catkin_ws/src/misssim/src"

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

tau = 0.1
max_step_count = 200

# Learning Setting
device = ('cuda'+':'+ str(gpu_num)) if torch.cuda.is_available() else 'cpu'

a_seed = 0
a_gamma = 0.9
max_step_count = 600

a_log_interval = 10
torch.manual_seed(a_seed)
np.random.seed(a_seed)
if device == ('cuda'+ ':' + str(gpu_num)) :
    torch.cuda.manual_seed_all(a_seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
RewardRecord = namedtuple('RewardRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

print(device)

# Learning Class Def

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc     = nn.Linear(5, 400)
        self.hd     = nn.Linear(400, 800)
        self.hd2     = nn.Linear(800, 400)
        #self.hd3     = nn.Linear(400, 200)
        self.mu_layer = nn.Linear(400, 2)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.hd.weight)   
        torch.nn.init.xavier_uniform_(self.hd2.weight)  
        #torch.nn.init.xavier_uniform_(self.hd3.weight)  
        torch.nn.init.xavier_uniform_(self.mu_layer.weight)

    def forward(self, s):
        s   = s.to(device)
        x   = (self.fc(s))
        x   = F.tanh(self.hd(x))
        x   = F.tanh(self.hd2(x))
        #x   = F.tanh(self.hd3(x))
        acc = self.mu_layer(x) #2 .0 * F.tanh(self.mu_layer(x))
        acc = acc.to('cpu')
        return acc

class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(7, 400)
        self.hd = nn.Linear(400, 800)
        self.hd2 = nn.Linear(800, 400)
        #self.hd3 = nn.Linear(400, 200)
        self.Q_layer = nn.Linear(400, 1)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.hd.weight)
        torch.nn.init.xavier_uniform_(self.hd2.weight)
        #torch.nn.init.xavier_uniform_(self.hd3.weight)
        torch.nn.init.xavier_uniform_(self.Q_layer.weight)

    def forward(self, s, a):
        s   = s.to(device)
        
        a   = a.to(device)
        x = self.fc(torch.cat([s, a], dim=1))

        x = F.tanh(self.hd(x))
        x = F.tanh(self.hd2(x))
        #x = F.tanh(self.hd3(x))
        state_value = self.Q_layer(x)
        state_value = state_value.to('cpu')
        return state_value


class Memory():

    memory_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.memory_pointer] = transition
        self.memory_pointer += 1
        if self.memory_pointer == self.capacity:
            self.memory_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():

    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 1
        self.eval_cnet, self.target_cnet = CriticNet().to(device).float(), CriticNet().to(device).float()
        self.eval_anet, self.target_anet = ActorNet().to(device).float(), ActorNet().to(device).float()
        self.memory = Memory(30000) #2000 #40000
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=crlr)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=aclr)

    def select_action(self, state):
        global mu_now
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.eval_anet(state)
        #mu_now = copy.deepcopy(mu.item())
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float))
        action = dist.sample()
        action = action.clamp(-99, 99)
        #print(action.detach().cpu().numpy()[0,:])
        return (action.detach().cpu().numpy()[0,:])

    def save_param(self, epsd):
        # pass
        torch.save(self.eval_anet.state_dict(), FilePath+'/params/anet_params_R,initL,Ohm'+str(epsd)+'.pkl')
        torch.save(self.eval_cnet.state_dict(), FilePath+'/params/cnet_params_R,initL,Ohm'+str(epsd)+'.pkl')

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1
        
        transitions = self.memory.sample(300)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            q_target = r + a_gamma * self.target_cnet(s_, self.target_anet(s_))
            
        q_eval = self.eval_cnet(s, a)
        
        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        with torch.no_grad():
            #lossset[1] = copy.deepcopy(c_loss.item())
            pass
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        with torch.no_grad():
            #lossset[0] = copy.deepcopy(a_loss.item())
            pass
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        for param, target_param in zip(self.eval_cnet.parameters(), self.target_cnet.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.eval_anet.parameters(), self.target_anet.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.var = max(self.var * 0.9999999, 0.01)

        return q_eval.mean().item()


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
#Learning_dat_basic.SessionDiscription = str(sys.argv[2])
Learning_dat_basic.SessionIdentifier = SessionIdentificationNumber

Learning_dat_basic.deltatime = dtSim
Learning_dat_basic.GPU = gpu_num

Learning_dat_basic.done = False
#-----------------------------------------------------------------------
CraftStreamPub = rospy.Publisher('/LearningInfo/Craft', LearningMonitor, queue_size=1)
MissiStreamPub = rospy.Publisher('/LearningInfo/Missile', LearningMonitor, queue_size=1)

#=======================================================================

aquiz_dat = rospy.Subscriber('/mavros/imu/data', Imu, processing_fnc)    # Garbage

def norm_R_reward(sacrifice):
    loged = -m.log10(-sacrifice)
    normV = loged/4
    return normV

def norm_A_reward(sacrifice,Vm,initLOS,t_f):
    stand = sacrifice/Vm/t_f/m.sqrt(abs(initLOS)+0.5)
    normV = ((-m.log(-stand))-3)/4
    return normV

# Instances +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MissileModel_1 = MissileEnv.Missile_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(MissileModel_1)
FighterModel_1 = AirCrftEnv.Craft_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(FighterModel_1)
MissileSeeker_1 = MissileEnv.Seeker(MissileModel_1, FighterModel_1)
FighterSeeker_1 = AirCrftEnv.Seeker(FighterModel_1, MissileModel_1)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


agent = Agent()

training_records    = []
Reward_records      = []

running_reward, running_q = -0.5, 0

#     __                          _                _____ __             __      
#    / /   ___  ____ __________  (_)___  ____ _   / ___// /_____ ______/ /______
#   / /   / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/   \__ \/ __/ __ `/ ___/ __/ ___/
#  / /___/  __/ /_/ / /  / / / / / / / / /_/ /   ___/ / /_/ /_/ / /  / /_(__  ) 
# /_____/\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /   /____/\__/\__,_/_/   \__/____/  
#                                     /____/                                    
ASCIIart.LearningStarts()
for i_ep in range(10000):

    t = 0. ##sec
    Cassette_tape       = []
    score = 0

    posM, hedM, spdM = init_picker()
    print('===================================================================================')
    print('EP : ', i_ep, '| Init : ','\n\t\t',posM,'\n\t\t',hedM,'\n\t\t',spdM)
    
    MissileSeeker_1.impactR = 50000
    MissileSeeker_1.RLdot = 1e-6
    closestR = 999999
    integral_acc = np.array([0.,0.])
    step_count = 0
    
    print('-------------------------------RL-------------------------------------------------')

    MissileModel_1.reset(posM, hedM, spdM, reset_flag = True)
    t = 0 ##sec
    step_count = 0
    MissileSeeker_1.impactR = 50000
    _, initLook, LOSdot, _, state = MissileSeeker_1.seek()

    energy_term = []
    energy = 0
    pngtime = []
    pngacc  = []
    print(MissileSeeker_1.Look.z, MissileSeeker_1.Look.y)

    while t<max_step_count:
        action = agent.select_action(state)
        Acc_cmd = Vector3(0, action[0], action[1])

        MissileModel_1.simulate(Acc_cmd)
        
        Rmag, Look, LOSdiot, _, state_ = MissileSeeker_1.seek()

        if gazebosim: Transmit2Gazebo(FighterModel_1, MissileModel_1)
        if Unitysim: Transmit2Unity(FighterModel_1, MissileModel_1)
                
        Mt_reward, Mm_reward, M_done, M_is_hit = MissileSeeker_1.spit_reward(Acc_cmd)
        Ct_reward, Cm_reward, C_done, C_is_hit = FighterSeeker_1.spit_reward(Vector3(0,0,0))
        
        if t == max_step_count-1:
            done = True

        if M_is_hit:
            #reward = reward + 2
            hitCount += 1
            print('\t'+'hit!!!++++++++++++++++++++++!!!!!!!!!!!!!!!!!'+str(hitCount))
            if not solved : 
                pass
        
        integral_acc += Mt_reward*dtSim
        Cassette_tape.append([state, np.array([Acc_cmd.y,Acc_cmd.z]), state_])

        state = state_
        
        if agent.memory.isfull:
            
            q = agent.update()
            running_q = 0.99 * running_q + 0.01 * q

        t = t + dtSim
        StreamDataFunnel("Missile", i_ep, t, MissileModel_1, Mt_reward, Mm_reward,M_done)
        StreamDataFunnel("Fighter", i_ep, t, FighterModel_1, Ct_reward, Cm_reward,C_done)
        
        
        if M_done:
            print(M_is_hit)
            break

    final_reward =    0.8*norm_R_reward(Mm_reward) \
                    + 0.1*norm_A_reward(integral_acc[0],spdM, initLook.y,t)\
                    + 0.1*norm_A_reward(integral_acc[1],spdM, initLook.z,t)
    score = final_reward
    
    for rowrow in Cassette_tape:
        agent.store_transition(Transition(rowrow[0], rowrow[1], score, rowrow[2]))
    if not slept:
        running_reward = score
    slept = True
    

    running_reward = running_reward * 0.9 + score * 0.1
    training_records.append(TrainingRecord(i_ep, running_reward))
    Reward_records.append(RewardRecord(i_ep, score))
    #print(i_ep, running_reward, running_q)
    if i_ep % a_log_interval == 0:
        print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
            i_ep, running_reward, running_q))
        
        plt.plot([r.ep for r in Reward_records], [r.reward for r in Reward_records])
        plt.title('RWDs')
        plt.xlabel('Episode')
        plt.ylabel('reward sum')
        plt.savefig(FilePath+"/img/reward.png")


        plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
        plt.title('RWDs-LPF')
        plt.xlabel('Episode')
        plt.ylabel('reward sum')
        plt.savefig(FilePath+"/img/reward_lpf.png")

        #a_render = True
    if M_is_hit:
        print("Solved! Running reward is now {}!".format(running_reward))
        #env.close()
        saveCount += 1
 
        agent.save_param(i_ep)

        if running_reward > 0: solved = True
        #with open('log/ddpg_training_records.pkl', 'wb') as f:
        #    pickle.dump(training_records, f)
        #break
    
    print('===================================================================================\n\n')

quit()
rospy.spin()  

#++++++++++++++++++++++++++++++Functions++++++++++++++++++++++++++++++

