# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        return np.matrix([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dt = params.dt
        q = params.q
        q1 = ((dt**3)/3)*q 
        q2 = ((dt**2)/2)*q 
        q3 = dt * q         
        return np.matrix([[q1, 0, 0, q2, 0, 0],
                         [0, q1, 0, 0, q2, 0],
                         [0, 0, q1, 0, 0, q2],
                         [q2, 0, 0, q3, 0, 0],
                         [0, q2, 0, 0, q3, 0],
                         [0, 0, q2, 0, 0, q3]])     
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############           
        #Update state estimates and error covariance
        X = self.F() @ track.x
        P = self.F() @ track.P @ np.transpose(self.F()) + self.Q()
        track.set_x(X)
        track.set_P(P)
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        K = track.P @ np.transpose(meas.sensor.get_H(track.x)) @ np.linalg.inv(self.S(track, meas, meas.sensor.get_H(track.x)))
        X = track.x + K @ self.gamma(track, meas)
        P = (np.identity(6) - K @ meas.sensor.get_H(track.x)) @ track.P
        track.set_x(X)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        return meas.z - meas.sensor.get_hx(track.x) 
        #use get_hx instead of get_H @ x because that correctly multiplies H and x for both lidar and camera (coordinate conversion done for camera)
        #if using get_H @ x, the states x are in the wrong frame (vehicle), not camera sensor frame
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        return meas.sensor.get_H(track.x) @ track.P @ np.transpose(meas.sensor.get_H(track.x)) + meas.R
        ############
        # END student code
        ############ 