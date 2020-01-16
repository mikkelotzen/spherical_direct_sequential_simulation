# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:37:33 2018

@author: rjoos
"""

import numpy as np
import h5py
from collections import defaultdict


# Outputs dictionary keys as strings
def swarmdata_strkey(l1=None,l2=None):
    swarm = h5py.File('saved_variables/data_10_2014-2016.mat','r') # load .mat file
    data=defaultdict(dict) # Prepare data dictionary
    if l1==None or l2==None: # No conditions on desired data
        for _,item in enumerate(swarm):
            for _,subitem in enumerate(swarm[item]):
                data[item][subitem] = np.array(swarm[item][subitem])
    elif l1!=None and l2!=None: # Only certain data is considered
        for _, item in enumerate(swarm):
            for i,comp1 in enumerate(l1):
                if item == comp1:
                    for _, subitem in enumerate(swarm[item]):
                        for j,comp2 in enumerate(l2):
                            if subitem == comp2:
                                data[item][subitem] = \
                                            np.array(swarm[item][subitem])
    return data

# Outputs dictionary keys as numbers
def swarmdata(l1=None,l2=None):
    swarm = h5py.File('saved_variables/data_10_2014-2016.mat','r') # load .mat file
    data=defaultdict(dict) # Prepare data dictionary
    if l1==None or l2==None: # No conditions on desired data
        for i,item in enumerate(swarm):
            for j,subitem in enumerate(swarm[item]):
                data[i][j] = np.array(swarm[item][subitem])
    elif l1!=None and l2!=None: # Only certain data is considered
        for i,item in enumerate(swarm):
            for n,comp1 in enumerate(l1):
                if item == comp1:
                    for j,subitem in enumerate(swarm[item]):
                        for m,comp2 in enumerate(l2):
                            if subitem == comp2:
                                data[n][m] = \
                                            np.array(swarm[item][subitem])
    return data

def swarmgrid(phi,theta,radius,cen,form,rad5):
    cen = tuple([i*np.pi/180 for i in cen])
    if rad5:
        radius = (radius+5)*np.pi/180
    else:
        radius*=np.pi/180
    if form == 'cap':
        eucli0 = np.sqrt((phi[0,:]-cen[0])**2+(theta[0,:]-cen[1])**2)
        # Find correct distances in coordinate grid
        trim0 = np.logical_and(eucli0 <= radius, eucli0 >= -radius)
        
        eucli1 = np.sqrt((phi[1,:]-cen[0])**2+(theta[1,:]-cen[1])**2)
        # Find correct distances in coordinate grid
        trim1 = np.logical_and(eucli1 <= radius, eucli1 >= -radius)
        trim = np.logical_or(trim0,trim1)
        return trim
    elif form == 'square':
        trim0 = np.logical_and(phi[0,:]-cen[0]<=radius, phi[0,:]-cen[0]>=-radius)
        trim1 = np.logical_and(theta[0,:]-cen[1]<=radius,theta[0,:]-cen[1]>=-radius)
        trim = np.logical_and(trim0,trim1)
        
        trim2 = np.logical_and(phi[1,:]-cen[0]<=radius, phi[1,:]-cen[0]>=-radius)
        trim3 = np.logical_and(theta[0,:]-cen[1]<=radius,theta[0,:]-cen[1]>=-radius)
        trimm = np.logical_and(trim2,trim3)
        
        trimfinal = np.logical_or(trim,trimm)
        return trimfinal
    else:
        print("Choose either form = 'cap' or form = 'square")
        return
        