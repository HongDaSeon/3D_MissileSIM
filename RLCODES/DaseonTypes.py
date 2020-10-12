#!/usr/bin/env python3
#-*- coding: utf-8 -*-


# Types Defined by Daseon

import PSpincalc as spin
from pyquaternion import Quaternion
import math as m
import numpy as np
import vpython as vp

class ASCIIart:
    def FDCLAB():
        print(   "\t\t\t   ____  ____   ___  \
                \n\t\t\t  (  __)(    \ / __) \
                \n\t\t\t   ) _)  ) D (( (__  \
                \n\t\t\t  (__)  (____/ \___) \
                \n\t\t\t   __     __   ____  \
                \n\t\t\t  (  )   / _\ (  _ \ \
                \n\t\t\t  / (_/\/    \ ) _ ( \
                \n\t\t\t  \____/\_/\_/(____/")

    def LearningStarts():
        print(   "    __                          _                _____ __             __ \
                \n   / /   ___  ____ __________  (_)___  ____ _   / ___// /_____ ______/ /______ \
                \n  / /   / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/   \__ \/ __/ __ `/ ___/ __/ ___/ \
                \n / /___/  __/ /_/ / /  / / / / / / / / /_/ /   ___/ / /_/ /_/ / /  / /_(__  )  \
                \n/_____/\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /   /____/\__/\__,_/_/   \__/____/   \
                \n                                    /____/                                     ")

    def VisualPython():
        print("  _    ___                  __   ____        __  __ \
              \n| |  / (_)______  ______ _/ /  / __ \__  __/ /_/ /_  ____  ____ \
              \n| | / / / ___/ / / / __ `/ /  / /_/ / / / / __/ __ \/ __ \/ __ \ \
              \n| |/ / (__  ) /_/ / /_/ / /  / ____/ /_/ / /_/ / / / /_/ / / / / \
              \n|___/_/____/\__,_/\__,_/_/  /_/    \__, /\__/_/ /_/\____/_/ /_/  \
              \n                                  /____/                        ")

    def Unity_ROS():
        print("      __  __      _ __              ____  ____  _____ \
                \n  / / / /___  (_) /___  __      / __ \/ __ \/ ___/\
                \n / / / / __ \/ / __/ / / /_____/ /_/ / / / /\__ \ \
                \n/ /_/ / / / / / /_/ /_/ /_____/ _, _/ /_/ /___/ / \
                \n\____/_/ /_/_/\__/\__, /     /_/ |_|\____//____/  \
                \n                 /____/                           ")

class Vector3:

    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def cast(lllist):
        return Vector3(lllist[0], lllist[1], lllist[2])

    def __repr__(self):
        return "Vector3({})".format(self.vec)

    def __add__(self, other):
        A = self.vec
        B = other.vec
        if len(A) != len(B):
            print("Error : Vector size"+str(len(A)+" and "+str(len(B)) + "missmatch" ))
            return None
        return Vector3(A[0]+B[0], A[1]+B[1], A[2]+B[2])

    def __sub__(self, other):
        A = self.vec
        B = other.vec
        if len(A) != len(B):
            print("Error : Vector size"+str(len(A)+" and "+str(len(B)) + "missmatch" ))
            return None
        return Vector3(A[0]-B[0], A[1]-B[1], A[2]-B[2])
        
    @property
    def vec(self):
        return np.array([self.x,self.y,self.z], dtype=np.float64)
    @vec.setter
    def vec(self, listvec):
        self.x = listvec[0]
        self.y = listvec[1]
        self.z = listvec[2]

    @property
    def VPvec(self):
        return vp.vec(self.x, self.y, self.z)

    @property
    def zyxvec(self):
        return np.array([self.z,self.y,self.x], dtype=np.float64)

    @property
    def mag(self):
        return np.sqrt(sum(self.vec**2))

    @property
    def direction(self):
        return Vector3.cast(self.vec / np.sqrt(sum(self.vec**2)))

    