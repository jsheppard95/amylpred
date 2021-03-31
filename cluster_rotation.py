#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:17:20 2018

@author: JacksonSheppard
"""
# Functions to rotate vectors in data set by average angles of clusters that
# form
# Way to process out rotation before sending data to neural network
# Last edit: 4/27/18

import numpy as np
import math
import random
from numpy import linalg
import matplotlib.pyplot as plt
from copy import deepcopy

# Defining functions to find angular average (theta and phi) of peptide unit
# vectors for a given frame
# Will then rotate all unit vectors by the found average theta and phi in each
# frame
# This causes the "poles" shown in VMD to be set by the laboratory frame
# This "processes out global rotation" from the neural net

##############################################################################

# Function to find average theta and phi of all unit vectors in a given frame:
# theta = arccos(z/r) = arccos(z), (r=1, unit vec); phi = arctan(y/x)

def cart_to_sphere(unit_vector):
    # Function to convert unit vector [x, y, z] into spherical coordinates [theta, phi]
    # Don't need r since unit vector -> r = 1
    theta = np.arccos(unit_vector[2])
    # Need to correct phi for points very close to z axis
    # If x ~ y, then arctan(y/x) ~ pi/4, not what we want going into the average
    # Therefore when x and y very close to 0, set phi = 0
    # Correct phi for points in quadrants II and III
    if unit_vector[0] == 0:
        if unit_vector[1] > 0:
            phi = np.pi/2
        elif unit_vector[1] < 0:
            phi = 3*np.pi/2
        elif unit_vector[1] == 0:
            phi = 0
    else:
        phi = np.arctan(unit_vector[1]/unit_vector[0]) # returns in range[-pi/2, pi/2]
        if (unit_vector[0] < 0) and (unit_vector[1] < 0): # x<0, y<0: computer QI -> sphere QIII
            phi = phi + np.pi
        elif (unit_vector[0] < 0) and (unit_vector[1] > 0): # x<0, y>0: computer QIV -> sphere QII
            phi = phi + np.pi
        elif (unit_vector[0] > 0) and (unit_vector[1] < 0): # x>0, y<0: computer -(phi) -> sphere +phi
            phi = phi + (2*np.pi)
    if (unit_vector[0] < 0) and (unit_vector[1] == 0):
        phi = phi + np.pi
    # Changing domain of phi to [-pi, pi], this line adjusts angles greater than pi properly
    if phi > np.pi:
        phi -= 2*np.pi
    angles = np.array([theta, phi])
    return angles

def find_avg_angles(frame):
    N = len(frame) # number of unit vectors
    theta_avg = 0
    phi_avg = 0
    for vec in frame:
        vec_sphere = cart_to_sphere(vec) # returns np.array([theta, phi])
        theta_avg += vec_sphere[0]
        phi_avg += vec_sphere[1] 
    theta_avg /= N
    phi_avg /= N
    return theta_avg, phi_avg
##############################################################################

# Function to rotate vectors using matrix multiplication
# rotation_matrix = [[cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)],
#                    [-sin(phi),              cos(phi),                 0],
#                    [sin(theta)cos(phi),  sin(theta)sin(phi), cos(theta)]]
# vec_prime = rotation_matrix*vec

def rotate_vectors_matrix(vectors, theta, phi):
    rotated_vectors = []
    rot_matrix = np.zeros((3,3))
    rot_matrix[0][0] = np.cos(theta)*np.cos(phi)
    rot_matrix[0][1] = np.cos(theta)*np.sin(phi)
    rot_matrix[0][2] = -np.sin(theta)
    rot_matrix[1][0] = -np.sin(phi)
    rot_matrix[1][1] = np.cos(phi)
    rot_matrix[1][2] = 0.0
    rot_matrix[2][0] = np.sin(theta)*np.cos(phi)
    rot_matrix[2][1] = np.sin(theta)*np.sin(phi)
    rot_matrix[2][2] = np.cos(theta)
    for vec in vectors:
        rotated_vec = rot_matrix.dot(vec)
        rotated_vectors.append(rotated_vec)
    return rotated_vectors
##############################################################################

# 4/17/18: Redesigning clustering function to work for arbitrary number of groups
# Using version of DBSCAN clustering algorithm, not accounting for outliers, all
# points will be assigned to a cluster
# "Neareast Neighbor search"

def remove_cluster_points(frame, cluster):
    # function to remove all points in cluster from the entire frame data set
    for vec in cluster:
        if vec in frame:
            frame.remove(vec)
    return frame

def initialize_cluster(frame, epsilon):
    # Function to initialize the first cluster given a frame and neighborhood distance
    # epsilon
    random_idx = np.random.randint(0, len(frame))
    vec_init = frame[random_idx]
    cluster = []
    cluster.append(vec_init)
    frame.pop(random_idx)
    for vec_i in frame:
        dist = np.sqrt( ((vec_init[0] - vec_i[0])**2) + ((vec_init[1] - vec_i[1])**2) + ((vec_init[2] - vec_i[2])**2) )
        if dist <= epsilon:
            cluster.append(vec_i)
    return cluster


def neighbor_cluster(frame):
    frame_l = [list(item) for item in frame]
    epsilon = .8 # max distance of neighborhood - will need to play with
    clusters = [] # list to hold each finished cluster of vectors
    # Initialize first cluster
    curr_cluster = initialize_cluster(frame_l, epsilon)
    # Remove points in cluster from data set    
    remove_cluster_points(frame_l, curr_cluster)
    # Begin adding vectors to curr_cluster by finding nearest neighbors of those already in cluster
    while True:
        # Iterate through each vector in current cluster and compare distance
        # to each remaining point in data set
        for i in range(len(curr_cluster)):
            for j in range(len(frame_l)):
                dist = np.sqrt( ((curr_cluster[i][0] - frame_l[j][0])**2) + ((curr_cluster[i][1] - frame_l[j][1])**2) + ((curr_cluster[i][2] - frame_l[j][2])**2) )
                if dist <= epsilon:
                    curr_cluster.append(frame_l[j])
            remove_cluster_points(frame_l, curr_cluster)
        # Finished with cluster, add it to clusters array and check if still more data to group
        clusters.append(curr_cluster)
        if len(frame_l) == 0:
            break
        # Initialize the next cluster
        curr_cluster = initialize_cluster(frame_l, epsilon)
        remove_cluster_points(frame_l, curr_cluster)
    return clusters

##############################################################################
# New function to rotate vectors in a frame -> fully vectorized version
# No longer rotating axes, now think of vectors rotating about fixed axis
# This function will rotate a vector by angle phi about z and then angle theta
# about x
# Have analytically solved for theta and phi such that this rotation can align
# the vector with the z-axis, will do so for cluster vector
# Last edit: 4/30/18
def get_theta_phi(vector):
    """
    This function will return the angular coordinates theta and phi used for
    rotation
    input: vector = np.array([x, y, z]) - should be a unit vector
    Returns: theta, phi derived analytically from rotating an arbitray vector
    into z-axis
    """
    a = vector[0]
    b = vector[1]
    c = vector[2]
    phi = np.arctan(-a/b)
    B = (-a*np.sin(phi)) + (b*np.cos(phi))
    theta = np.arcsin( (( ((B+c)**2) + ((c-B)**2) )**(-1/2)) ) - np.arctan2(B+c, c-B)
    theta_check = float(theta)
    phi_check = float(phi)
    if (math.isnan(theta_check) == True) or (math.isnan(phi_check) == True):
        print('bad vector:', vector)
    return theta, phi

def vector_rotation(vector, theta, phi):
    """
    This function will rotate a vector by theta and phi using the rotation
    matrix derived analytically
    input: vector = unit vector np.array([x, y, z])
           phi = rotation about z-axis
           theta = rotation about x-axis (sign of rotation defined by
                   right-hand-rule)
    output: rotated_vector = np.array([x', y', z'])
    """
    theta_check = float(theta)
    phi_check = float(phi)
    if (math.isnan(theta_check) == True) or (math.isnan(phi_check) == True):
        return np.array([0, 0, 1])
    rot_matrix = np.zeros((3,3))  # access elements as rot_matrix[row][column]
    rot_matrix[0][0] = np.cos(phi)
    rot_matrix[0][1] = np.sin(phi)
    rot_matrix[0][2] = 0
    rot_matrix[1][0] = -np.cos(theta)*np.sin(phi)
    rot_matrix[1][1] = np.cos(theta)*np.cos(phi)
    rot_matrix[1][2] = np.sin(theta)
    rot_matrix[2][0] = np.sin(theta)*np.sin(phi)
    rot_matrix[2][1] = -np.sin(theta)*np.cos(phi)
    rot_matrix[2][2] = np.cos(theta)
    rot_vector = rot_matrix.dot(vector)
    return rot_vector

def rotate_ensemble(frame):
    """
    This function will rotate each vector in a frame such that the largest
    cluster of vectors will align with the z axis
    input: frame = list of numpy arrays, each corresponding to a peptide
    output: rotated_frame = new list of numpy arrays after coordinate
                            transformation
    """
    # First get clusters
    clusters = neighbor_cluster(frame)
    # Only want to rotate if neighbor_cluster finds under a max number of clusters
    if len(clusters) > 6:
        rotated_frame = deepcopy(frame)
    else:
        # Want to align largest cluster with z-axis: find largest cluster
        rotated_frame = []
        biggest_cluster = clusters[0]
        for cluster in clusters:
            if len(cluster) > len(biggest_cluster):
                biggest_cluster = cluster
        # Average coordinates in biggest_cluster:
        x_av = 0
        y_av = 0
        z_av = 0
        for vec in biggest_cluster:
            x_av += vec[0]
            y_av += vec[1]
            z_av += vec[2]
        N = len(biggest_cluster)
        x_av /= N
        y_av /= N
        z_av /= N
        # Need to normalize avg_vec:
        mag = np.sqrt( (x_av**2) + (y_av**2) + (z_av**2) )
        x_av /= mag
        y_av /= mag
        z_av /= mag
        # Now can create unit vector [x_av, y_av, z_av]
        avg_vec = np.array([x_av, y_av, z_av])
        theta, phi = get_theta_phi(avg_vec)
        for vec in frame:
            rotated_vec = vector_rotation(vec, theta, phi)
            rotated_frame.append(rotated_vec)
    return rotated_frame

def rotate_trajec(frames):
    """
    Function to rotate each frame in full trajectory
    Actual function used at command line
    """
    rotated_frames = []
    for frame in frames:
        rotated_frame = rotate_ensemble(frame)
        rotated_frames.append(rotated_frame)
    return rotated_frames
##############################################################################

# Testing Functions:

##############################################################################

def rotation_test_single(frame):
    groups = neighbor_cluster(frame)
    thetas = []
    phis = []
    for group in groups:
        theta_i, phi_i = find_avg_angles(group)
        thetas.append(theta_i)
        phis.append(phi_i)
    theta = min(thetas)
    idx = thetas.index(theta)
    phi = phis[idx]
    rotated_frame = rotate_vectors_matrix(frame, theta, phi)
    return rotated_frame

##############################################################################

# 4/17/18
# Function to test rotation matrix for angles theta and phi;
# Given a unit vector with spherical angles theta and phi, multiplying
# the rotation matrix(theta, phi) by the unit vector should return a vector
# (0, 0, 1)
# Works successfully 
def rotation_matrix_test():
    unit_vec = np.array([1, 0, 0]) # shape(3,) -> (x, y, z)
    angles = cart_to_sphere(unit_vec)
    theta = angles[0]
    phi = angles[1]
    rot_matrix = np.zeros((3,3))
    rot_matrix[0][0] = np.cos(theta)*np.cos(phi)
    rot_matrix[0][1] = np.cos(theta)*np.sin(phi)
    rot_matrix[0][2] = -np.sin(theta)
    rot_matrix[1][0] = -np.sin(phi)
    rot_matrix[1][1] = np.cos(phi)
    rot_matrix[1][2] = 0.0
    rot_matrix[2][0] = np.sin(theta)*np.cos(phi)
    rot_matrix[2][1] = np.sin(theta)*np.sin(phi)
    rot_matrix[2][2] = np.cos(theta)
    vec_prime = rot_matrix.dot(unit_vec) # shape(3,) -> (x', y', z')
    return vec_prime

##############################################################################

# Function to test neighbor_cluster method:
# Will iterate through each frame and determine number of clusters
# If number of clusters over a certain threshold, will not rotate frame
# If number of cluster under threshold, rotating to avg theta and phi of clusteer
# with most points
def rotation_neighbors_test(frames):
    rotated_frames = []
    for frame in frames:
        clusters = neighbor_cluster(frame)
        if len(clusters) > 6:
            rotated_frames.append(frame)
        else:
            biggest_cluster = clusters[0]
            for cluster in clusters:
                if len(cluster) > len(biggest_cluster):
                    biggest_cluster = cluster
            theta, phi = find_avg_angles(biggest_cluster)
            rotated_frame = rotate_vectors_matrix(frame, theta, phi)
            rotated_frames.append(rotated_frame)
    return rotated_frames

##############################################################################
def outputThetaPhi(frames):
    """
    This function will iterate through the frames in the original (non-rotated)
    trajectory and output Theta and Phi for the average vector of each frame into 
    a text file.
    (theta, phi) = (0, 0) -> No rotation performed
    Angles given in radians
    """
    angles = [] # list containing pair [theta, phi] for each frame from 0 to len(frames)-1
    for frame in frames:
        # First get clusters
        clusters = neighbor_cluster(frame)
        # Only want to rotate if neighbor_cluster finds under a max number of clusters
        if len(clusters) > 6:
            angle = [0, 0]
            angles.append(angle)
        else:
            # Find biggest cluster:
            biggest_cluster = clusters[0]
            for cluster in clusters:
                if len(cluster) > len(biggest_cluster):
                    biggest_cluster = cluster
            # Average coordinates in biggest_cluster:
            x_av = 0
            y_av = 0
            z_av = 0
            for vec in biggest_cluster:
                x_av += vec[0]
                y_av += vec[1]
                z_av += vec[2]
            N = len(biggest_cluster)
            x_av /= N
            y_av /= N
            z_av /= N
            # Need to normalize avg_vec:
            mag = np.sqrt( (x_av**2) + (y_av**2) + (z_av**2) )
            x_av /= mag
            y_av /= mag
            z_av /= mag
            # Now can create unit vector [x_av, y_av, z_av]
            avg_vec = np.array([x_av, y_av, z_av])
            theta, phi = get_theta_phi(avg_vec)
            angle = [theta, phi]
            angles.append(angle)
    return angles
     