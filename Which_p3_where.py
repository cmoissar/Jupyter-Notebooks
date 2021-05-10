#!/usr/bin/env python
# coding: utf-8


import netCDF4 as nc
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
from os import listdir
from os.path import isfile, join

# /!\ hardcoded
ny = 720
nz = 660
dx = 1

ymin = 80
ymax = 90

zmin = -5
zmax = 5

def simpos(pos, npos, dpos):
    return pos/dpos + npos/2

simymin = simpos(ymin, ny, dx)
simymax = simpos(ymax, ny, dx)

simzmin = simpos(zmin, nz, dx)
simzmax = simpos(zmax, nz, dx)

print(simymin, simymax)
print(simzmin, simzmax)

i = 0
relevant_p3_files = []

for p3_file in listdir('./'):

    if not(".nc" in p3_file):
        continue

    p3_data = nc.Dataset(p3_file,'r')
    posy = np.array(p3_data.variables['particule_y'])
    posz = np.array(p3_data.variables['particule_z'])

    i+=1
    if (   (all(posy > simymax) or all(posy < simymin)) 
        or (all(posz > simzmax) or all(posz < simzmin)) ):
        print("file", i, "/7199: nope")
    else:
        print(p3_file, "is good")
        relevant_p3_files.append(p3_file)

#    posx = p3_data.variables['particule_x']
#    posy = p3_data.variables['particule_y']
#    posz = p3_data.variables['particule_z']

#    if any( posy_min < posy < posy_max ):
#        if any( posz_min < poz < posz_max ):
            
#            for particle in p3_data.variables:

#                if isincube(particle):

#                   group.append( { 'id': p3_data_variables['particule_id'],
#                                   'posx': p3_data.variables['particule_x'],
#                                   'posy': p3_data.variables['particule_y'],
#                                   'posz': p3_data.variables['particule_z'] } )





#velx = p3_data.variables['particule_vx']
#vely = p3_data.variables['particule_vy']
#velz = p3_data.variables['particule_vz']



#rcParams["figure.figsize"] = [20, 15]
#plt.scatter(np.array(posx), np.array(velx), s = 0.01)
#plt.scatter(np.array(posx), np.array(posy), s = 0.01)

