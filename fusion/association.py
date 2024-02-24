# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        N = len(track_list)
        M = len(meas_list)
        self.association_matrix = np.inf*np.ones((N,M)) # reset matrix
        self.unassigned_tracks = list(range(N)) #zero based indexing
        self.unassigned_meas = list(range(M)) #zero based indexing
        
        for i in range (N):
            track = track_list[i] #6 states in each [track.x]
            for j in range (M):
                nz = len(meas_list[0].z)
                meas = meas_list[j] #3 measurements in each [meas.z] for lidar, 2 for camera
                dist = self.MHD(track, meas, KF) #a scalar value
                if self.gating(dist, nz):
                    self.association_matrix[i,j] = dist
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self, original_matrix):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        #row, col
        min_value_index = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        if self.association_matrix[min_value_index] == np.inf:
            return np.nan, np.nan
        #have to check original association matrix to get correct track and measurement ID, 
        #after the matrix shrinks, the smallest value will be at a different index not corresponding to the correct track and measurement
        track_meas_indices = np.where(original_matrix == self.association_matrix[min_value_index])        
        update_track = track_meas_indices[0].item()
        update_meas = track_meas_indices[1].item()     
        #delete row (axis = 0)
        self.association_matrix = np.delete(self.association_matrix, min_value_index[0], 0)
        #delete column (axis = 1)
        self.association_matrix = np.delete(self.association_matrix, min_value_index[1], 1)        
        # remove track / measurement pair from list, anything that remains will be unassigned
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, nz): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        conf = params.gating_threshold #99.5% confidence 
        G = chi2.ppf(conf, nz) #gating threshold
        if MHD <= G:
            return True
        return False
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        H = meas.sensor.get_H(track.x)
        S = H*track.P*H.transpose() + meas.R
        MHD = np.transpose(KF.gamma(track, meas))*np.linalg.inv(S)*KF.gamma(track, meas)
        return MHD
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
        original_matrix = np.copy(self.association_matrix)
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas(original_matrix)
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management (new tracks created here for any unassigned measurements)
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)