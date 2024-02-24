# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############
        
        dim = params.dim_state
        self.x = np.zeros((dim,1))
        self.P = np.zeros((dim,dim))
        
        z = np.array([meas.z[0],meas.z[1],meas.z[2]])
        z_to_veh = M_rot @ z
        
        self.x[0] = z_to_veh[0]
        self.x[1] = z_to_veh[1]
        self.x[2] = z_to_veh[2]

        P = M_rot @ meas.R @ np.transpose(M_rot)
        self.P[0:3,0:3] = P
        self.P[3,3] = params.sigma_p44
        self.P[4,4] = params.sigma_p55
        self.P[5,5] = params.sigma_p66
        self.state = 'initialized'
        self.score = 1/params.window
        
        ############
        # END student code
        ############ 
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        
        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    print("track score decreasing from", track.score)
                    track.score = track.score - 0.1 

        # delete old tracks based on their state and score 
        threshold = params.delete_threshold
        Pmax = params.max_P
        for track in self.track_list:
            if track.state == 'confirmed' and track.score < threshold:
                print("deleting confirmed", track.score)
                self.delete_track(track)
            if track.state == 'initialized' or track.state == 'tentative':
                #delete track if score too low or if x, y covariance exceeds threshold
                if track.score < 0.1 or track.P[0,0] > Pmax or track.P[1,1] > Pmax:
                    print("deleting init or tentative", track.score)
                    self.delete_track(track)

        ############
        # END student code
        ############ 
            
        # initialize new track with unassigned measurement 
        #(e.g: at timestep 0 there's 2 measurements but no tracks) - association matrix is skipped
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############
        
        cthres = params.confirmed_threshold
        #must cap off track score at a value above confirmed threshold
        if track.score < cthres:
            track.score = track.score + 0.1
            if track.score > cthres:
                track.state = 'confirmed'
            else:
                track.state = 'tentative'

        ############
        # END student code
        ############ 