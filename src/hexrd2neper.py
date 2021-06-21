# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 08:19:34 2016

@author: Armand Beaudoin

"""
#%%

import math

import matplotlib
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

from hexrd import matrixutil as mutil

from hexrd.rotations import quatOfExpMap, rotMatOfExpMap

import time

#%%
#######################################################################################################
#
# LOAD IN THE DATA
#
# The data contained in the grains.out file are as follows:
# 
#    0   grain ID         
#    1   completeness                 
#    2   chi2                              goodness of fit, looking for            
#  3-5   xi[0], xi[1], xi[2]               orientation in exp. map format; 
# .                                        axis scaled by rotation angle
# 
#  6-8   tVec_c[0], tVec_c[1], tVec_c[2]   center of mass in sample frame [mm]
#  
#                                          inverse stretch tensor components
#  9-14  vInv_s[0], vInv_s[1], vInv_s[2], vInv_s[3]*sqrt(2), vInv_s[4]*sqrt(2),vInv_s[5]*sqrt(2)
# 
#                                          strain components
# 15-20  ln(V[0,0]), ln(V[1,1]), ln(V[2,2]), ln(V[1,2]), ln(V[0,2]), ln(V[0,1])


in_dir = 'data/'
scan_name = in_dir + 'grains_Ti7Al_creep_0020.out'
    
grainsOut = np.genfromtxt(scan_name,comments='#')

#%%
#% Write out coordinates for use by neper

out_dir = 'data/'
crd_file = open(out_dir + 'coords_0020.dat','w')
     
#for i in range(0,grainsOut.shape[1]):
#    crd_file.write( \
#                   '{:12.4e},{:12.4e},{:12.4e}\n'.format( \
#                    grainsOut[1,i,6], grainsOut[1,i,7], grainsOut[1,i,8] ) )

iSubset = (grainsOut[:,1]>0.5) & (grainsOut[:,2]<0.005)
grainsSubset = grainsOut[iSubset,:]

print('geometry limits')
print( grainsSubset[:,6].min(), grainsSubset[:,6].max() )
print( grainsSubset[:,7].min(), grainsSubset[:,7].max() )
print( grainsSubset[:,8].min(), grainsSubset[:,8].max() )


for i in range(0,grainsSubset.shape[0]):
    crd_file.write( \
                   '{:14.8g} {:14.8g} {:14.8g}\n'.format( \
                    grainsSubset[i,6], grainsSubset[i,7], grainsSubset[i,8] ) )
crd_file.close()
 
#%%
# Write out data for finite element simulations


# Rotations
rotf_name = out_dir + "rotations_0020.dat"

rot_file = open(rotf_name,'w')

exp_maps_step = np.atleast_2d(grainsSubset[:, 3:6])
rMats_subset = rotMatOfExpMap(exp_maps_step.T)
    
for i in range(0,grainsSubset.shape[0]):
    rot_file.write( \
                   '{:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g}\n'.format( \
                rMats_subset[i,0,0], rMats_subset[i,0,1], rMats_subset[i,0,2], \
                rMats_subset[i,1,0], rMats_subset[i,1,1], rMats_subset[i,1,2], \
                rMats_subset[i,2,0], rMats_subset[i,2,1], rMats_subset[i,2,2] ) )                        
rot_file.close()

# Lattice strains
strainf_name = out_dir + "strains_0020.dat" 
strain_file = open(strainf_name,'w')
   
for i in range(0,grainsSubset.shape[0]):
# 15-20  ln(V[0,0]), ln(V[1,1]), ln(V[2,2]), ln(V[1,2]), ln(V[0,2]), ln(V[0,1])
    strain_file.write( \
                   '{:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g} {:14.8g}\n'.format( \
                    grainsSubset[i,15], grainsSubset[i,20], grainsSubset[i,19], \
                    grainsSubset[i,20], grainsSubset[i,16], grainsSubset[i,18],
                    grainsSubset[i,19], grainsSubset[i,18], grainsSubset[i,17] ) )
strain_file.close()
