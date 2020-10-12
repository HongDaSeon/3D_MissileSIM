#!/usr/bin/env python3
#-*- coding: utf-8 -*-
################################################################################################################################################################################ -ohh`  #####################                
#                                                                                                                                                                             -ohNMMMMd`                    #
#                                                                                                                                                                         -ohNMMMMMmMMMd`                   #
#                                                                                                                                                                     -ohNMMMMNho-  yMMMd`                  #
#                                                                                                                                                                 -ohNMMMMNho-       yMMMd`                 #
#                                                                                                                                                             -ohNMMMMNho-            yMMMd`                #
#                                                                                                                                                         -+hNMMMMMdo-                 yMMMd`               #
#                                                                                                                                                     -+hNMMMMMdo:`                     yMMMd`              #
#                                                                                                                                                 -+hNMMMMMho-                           sMMMd`             #
#                                                                                                                                                `dMMMMho-                                sMMMd`            #
#                                                                                                                                                 `dMMMo                                   sMMMd`           #
#                                                                                                                                                  `dMMMs                                   sMMMd`          #
#                                                                                                                                                   `dMMMs                                   sMMMm`         #
#                                                                                                                                                    `dMMMo         `:osyyo/.                 sMMMm`        #
#                                                                                                                                                     `dMMMo      `sNMMMMMMMMh-                sMMMm`       #
#    MMMMMMMMo mMMMMNNds-    -yNMMMMNd-      sMMs                -MMh                                    `ddh                                          `dMMMo    `mMMMMMMMMMMMM/                sMMMm`      #
#    MMM/----. mMMo--+mMMy  oMMN+.`./h:      sMMy      -+osso/.  -MMh-+so:    :osso/`  .++:`/+`-+osso/. `+MMN++/  -+sso/.  `++/`:+/++-   :++.           `dMMMo   oMMMMMMMMMMMMMm                 sMMMm.     #
#    MMMyssss` mMM/   `mMM/.MMM-             sMMy      oysosmMM/ -MMNhydMMy -mMNsodMN/ /MMNMNN.oysosmMM/.yMMMyyo`dMMyohMMo .MMNNNN+NMN. -MMd             `dMMMs  oMMMMMMMMMMMMMm              ./ymMMMMm`    #
#    MMMhyyyy` mMM/    dMMo-MMM`             sMMy      .+syhmMMs -MMh   NMM.hMM/  `MMM`/MMh    .+syhmMMs `MMN   sMMo   mMM-.MMm`   :MMh dMN.              `dMMMs `mMMMMMMMMMMMM:          `/ymMMMMMms/`     #
#    MMM.      mMM/  `oMMm` mMMy`    /.      sMMy     :MMd:-oMMs -MMh  `NMM`hMM/  `MMN`/MMy   :MMd:-oMMs `MMN   oMMs   mMM-.MMm     oMMdMM:                `dMMMs `sNMMMMMMMMh-       `/smMMMMMms/`         #
#    MMM.      mMMmdmMMNs`  `yMMNdhdNM:      sMMNmmmmd-MMNssmMMs -MMNyyNMN/ .mMNysmMN/ /MMy   -MMNssmMMs  mMMhss`hMMysdMMo .MMm      hMMMs                  `dMMMs   -+ssso/`     `/smMMMMMms/`             #
#    ///`      //////:.       `:+oo+:.       -//////// .+o+:.//- `//::++/`    -+oo+:`  .//-    .+o+:.//-   :+o+:  ./ooo/`  `///      +MMd                    `dMMMs           ./smMMMMMms/`                 #
#                                                                                                                                   .NMN.                     `dMMMs      ./ymMMMMMms/`                     #
#                                                                                                                                   `..`                       `hMMMs `/smMMMMMmy/`                         #
#                                                                                                                                                               `hMMMmMMMMMmy/.                             #
#                                             T  H  E    M  O  T  I  O  N    T  E  C  H  N  O  L  O  G  Y    I  N  N  O  V  A  T  I  O  N  S                     `hMMMMmy/.                                 #
#                                                                                                                                                                 `yy/.                                     #
#                                                                                                                                                                                                           #
######################################################################################## /+:` ###############################################################################################################

# Pseudo 5 DOF 3dimensional Aircraft_Environment for Reinforcement Learning
#   Version --proto
#   Created by Hong Daseon

import rospy
import PSpincalc as spin
from pyquaternion import Quaternion
from DaseonTypes import Vector3
import math as m
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time
import copy
import pdb

#AirCraft coordinate trans Model
Debug = False
gAcc = 0. #9.806

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


class Craft_3D:
    
    def __init__(self, scavel, initPos, initAtt, dt):
        self.initval        = [scavel, initPos.x, initPos.y, initPos.z, initAtt.z, initAtt.y, dt]
        self.scavel         = scavel

        self.pos            = initPos
        self.att            = initAtt

        self.acc            = Vector3(0.,0.,0.)
        self.dpos           = Vector3(0.,0.,0.)
        self.datt           = Vector3(0.,0.,0.)

        self.dt             = dt
        
        self.Qnb            = RotationQuaternion(self.att)
      
        self.reset_flag     = True


    def simulate(self, _acc):
        self.acc          = _acc

        self.datt.z         = self.acc.y / (self.scavel * m.cos(self.att.y))
        self.datt.y         = - ( self.acc.z + gAcc * m.cos(self.att.y) ) / self.scavel
        self.datt.x         = 0.

        self.att.z          = self.att.z + self.datt.z * self.dt
        self.att.y          = self.att.y + self.datt.y * self.dt
        self.att.x          = self.att.x + self.datt.x * self.dt
        
        self.Qnb            = RotationQuaternion(self.att)

        posRatee            = self.Qnb.rotate([self.scavel, 0, 0])
        self.dpos.x         = posRatee[0]
        self.dpos.y         = posRatee[1]
        self.dpos.z         = posRatee[2]

        self.pos.x          = self.pos.x + self.dpos.x*self.dt
        self.pos.y          = self.pos.y + self.dpos.y*self.dt
        self.pos.z          = self.pos.z + self.dpos.z*self.dt
        
        return self.dpos, self.pos

    def reset(self, _pos, _att, Vm, reset_flag):
        self.scavel         = Vm

        self.pos            = _pos

        self.att            = _att

        self.acc            = Vector3(0.,0.,0.)

        self.dt             = self.dt
        
        self.Qnb            = RotationQuaternion(self.att)

        posRatee            = self.Qnb.rotate([self.scavel, 0, 0])
        self.dpos.x         = posRatee[0]
        self.dpos.y         = posRatee[1]
        self.dpos.z         = posRatee[2]
        
        #print('just after reset : ',self.Qnb)
        self.reset_flag     = reset_flag
    
    def __str__(self):
        nowpos = 'x : '+ format(self.pos.x,".2f")+ ' y : '+ format(self.pos.y,".2f")+ ' z : '+ format(self.pos.z,".2f")

        return nowpos

class Seeker:
    #prevR = 0
    def __init__(self, Fighter, Target):
        self.Rvec       = Target.pos - Fighter.pos
        self.Vvec       = Target.dpos - Fighter.dpos

        self.direcVec   = Fighter.dpos

        self.Target     = Target
        self.Fighter    = Fighter
        
        self.impactR    = 9999999

        LOSz, LOSy      = self.azimNelev(self.direcVec)
        self.LOS        = Vector3(0.,LOSy, LOSz)
        self.dLOS       = Vector3(0., 0., 0.) # body frame

        Lookz, Looky    = self.azimNelev(self.Rvec - self.direcVec)
        self.Look       = Vector3(0., Looky, Lookz)

        self.firstrun   = True

        self.prev_Rm  = Vector3(9999999, 9999999, 9999999)

        self.t2go       = 600

    def angle(self, vec1, vec2):
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]      # dot product
        det = vec1[0]*vec2[1] - vec2[0]*vec1[1]      # determinant
        return m.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    def azimNelev(self, vec):
        azim = m.atan2( vec.y, vec.x)
        elev = m.atan2( -vec.z, m.sqrt( vec.x**2 + vec.y**2))
        return azim, elev

    def seek(self):
        def normLd(LOSdotval):
            return (LOSdotval)*1000
        
        def normVm(Vval):
            return (Vval-400)/200
        #pdb.set_trace()
        self.t2go       = 0
        
        self.Rvec       = self.Target.pos - self.Fighter.pos
        self.Vvec       = self.Target.dpos - self.Fighter.dpos
        self.direcVec   = self.Fighter.dpos
                
        LOSz, LOSy      = self.azimNelev(self.direcVec)
        self.LOS        = Vector3(0.,LOSy, LOSz)

        Lookz, Looky    = self.azimNelev(Vector3.cast(self.Fighter.Qnb.inverse.rotate(self.Rvec.vec)))
        self.Look       = Vector3(0., Looky, Lookz)
        
        RjxVj = np.cross(self.Rvec.vec, self.Vvec.vec)
        RjdRj = np.dot(self.Rvec.vec, self.Rvec.vec)
        Ldotn = RjxVj/RjdRj
        
        Ldotb = self.Fighter.Qnb.inverse.rotate(Ldotn)
        self.dLOS = Vector3.cast(Ldotb)
        self.Fighter.reset_flag = False
        return self.Rvec.mag, self.Look, self.dLOS, self.Fighter.scavel,\
                                                                    np.array([  normVm(self.Vvec.x),\
                                                                                normVm(self.Vvec.y),\
                                                                                normVm(self.Vvec.z),\
                                                                                normLd(self.dLOS.y),\
                                                                                normLd(self.dLOS.z) ] ) 

    def spit_reward(self, acc):
        
        OOR         = (self.Look.y < -1.57)|(self.Look.y > 1.57)|(self.Look.z < -1.57)|(self.Look.z > 1.57) # Out of range
        if OOR:
            Rf_1 = self.prev_Rm
            Rf = self.Fighter.pos
            
            R3 = Rf - Rf_1
            A = R3
            B = (self.Target.pos - Rf_1) - R3
            
            if self.Rvec.mag < 50:

                self.impactR = (Vector3.cast(np.cross(A.vec,B.vec)).mag) / A.mag 

            else:
                self.impactR = self.Rvec.mag

            print('impacR : ', self.impactR)
            
            if Debug : pdb.set_trace()
        else:
            self.prev_Rm = copy.deepcopy(self.Fighter.pos)


        hit         = (self.impactR <2)

        step_reward  = - (acc.y**2) - (acc.z**2)
         #0.01*-Rdot - 3*abs(self.LOS) - 1.2*abs(self.LOS)*500*abs(Ldot) + (2/(self.R / 8000))**2.5 - (self.R<10000)*self.R/5000 #-1000*abs(Ldot)   -self.R/10000  # - self.R/100
        
        mc_reward    =  - (self.impactR)
        #reward = (not OOR)*reward -OOR*25
        
        return step_reward, mc_reward, (OOR), hit

