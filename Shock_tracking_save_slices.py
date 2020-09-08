#!/usr/bin/env python
# coding: utf-8

# ## Import modules and functions

# In[1]:


# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import glob
import re
import os

import pylab as pl
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from scipy.signal import savgol_filter

from matplotlib.gridspec    import GridSpec
import import_ipynb

import Module_Diagnostics as MD
import numpy as np
from tempfile import mkdtemp
import os.path as path
import sys

from pathlib import Path
import json

#Debugger. For some reason, using it inside a function works well. Otherwise...
from IPython.core.debugger import set_trace
#exemple: 
# def debug():
#     set_trace()
    
#     `code_to_debug`
    
#     return

# debug()


# ## Plot parameters

# In[2]:


# %matplotlib notebook
rcParams["figure.figsize"] = [9.4, 4.8]
# matplotlib.use('nbagg') #_comment this line if you don't need to interact with plots (zoom, translations, savings...)


# ## Choose run and time for analysis

# In[3]:


run_name = 'RUN_NAME'

### Only if working on lx-moissard
Cluster = 'Occ/'
run_name = '20_08_18_new_big_one_0'
filepath = '../ncfiles/'

#This is used by the functions find_ip_shock(N, V) and find_mc_leading_edge(B)
metadata = {'t_shock_entrance' : 130,
            't_shock_exit'     : 240,
            't_MC_entrance'    : 130,
            't_MC_exit'        : 270}
#todo: autodefine t_collision? maybe from story_reader will be easier, as lines will cross on the multivariate plot

from_time = 226
to_time = 240 #metadata['t_shock_exit']

date = re.search('Magw_(.+?)_t', glob.glob(filepath+'Magw*_t'+ '%05d' % from_time +'.nc')[0]).group(1) 

print(f'date of the simulation (DD_MM_YY): {date}')


# In[4]:


# Prepare for plt.savefig
storing_directory = filepath + "../shock_tracking/"
path_png = Path(storing_directory)
if path_png.exists():
    pass
else:
    path_png.mkdir()


# In[5]:


storing_directory_json = filepath + "../shock_tracking/"

path_store_json = Path(storing_directory_json)

if not(path_store_json.exists()):
    os.system(f'mkdir {path_store_json}')

name = "shock_tracking_" + run_name + ".json"
path_json = Path(storing_directory_json + name)


# ## Get data in Hsw, Magw and Elew

# In[6]:


def collect_slices(data, Hsw):
    
    x = np.array(np.around(Hsw['x']))
    y = np.array(np.around(Hsw['y']))
    z = np.array(np.around(Hsw['z']))

    nx,  ny,  nz  = len(x), len(y), len(z)
    # Location of the planet is defined in the .ncfiles as (x,y,z) = (0,0,0)
    # Location of the planet is defined in the .ncfiles as (x,y,z) = (0,0,0)
    nx0, ny0, nz0 = ( int(np.where(abs(x)==min(abs(x)))[0]),
                      int(np.where(abs(y)==min(abs(y)))[0]), 
                      int(np.where(abs(z)==min(abs(z)))[0])  )
    
    result = {}
        
    for item in data:
        
        list_xy = {}
        list_xz = {}
        print(item)
        
        list_relevant_y = [-300, -100, -90, -80, -70, 0, 70, 80, 90, 100, 300]
        list_iy = [np.where(y == r_y)[0][0] for r_y in list_relevant_y]
        for iy in list_iy:
            list_xy.update({'y = ' + str(y[iy]): [float(value) for value in ( Hsw[item][:, iy-1 , nz0]
                                                                             +Hsw[item][:, iy   , nz0]
                                                                             +Hsw[item][:, iy+1 , nz0] )/3] })
            
        list_relevant_z = [-300, -100, -90, -80, -70, 0, 70, 80, 90, 100, 300]
        list_iz = [np.where(z == r_z)[0][0] for r_z in list_relevant_z]
        for iz in list_iz:
            list_xz.update({'z = ' + str(z[iz]): [float(value) for value in ( Hsw[item][:, ny0 , iz-1]
                                                                             +Hsw[item][:, ny0 , iz  ]
                                                                             +Hsw[item][:, ny0 , iz+1] )/3] })
        
        print(f'mean value for {item} is {1./2*(np.nanmean(list(list_xy.values())) + np.nanmean(list(list_xz.values())))}')
        result.update({  item: { '(xy) plane': list_xy,
                                 '(xz) plane': list_xz }  })
    return result


# ### Upload V

# In[7]:


for time in range(from_time, to_time):
    time = '%05d' % time    # Change the time to string format, needed by functions

    ## Load Vxyz
    Hsw = MD.import_data_3D(filepath, date, time, 'Hsw')
  
    x = np.array(np.around(Hsw['x']))
    y = np.array(np.around(Hsw['y']))
    z = np.array(np.around(Hsw['z']))

    cwp = Hsw['c_omegapi']
    gstep = Hsw['gstep']

    nx,  ny,  nz  = len(x), len(y), len(z)
    # Location of the planet is defined in the .ncfiles as (x,y,z) = (0,0,0)
    # Location of the planet is defined in the .ncfiles as (x,y,z) = (0,0,0)
    nx0, ny0, nz0 = (  int(np.where(abs(x)==min(abs(x)))[0]),
                       int(np.where(abs(y)==min(abs(y)))[0]), 
                       int(np.where(abs(z)==min(abs(z)))[0])  )

      
    new_data = {  time: collect_slices(['Vx', 'Vy', 'Vz'], Hsw), 'x': [float(xj) for xj in x],
                                                                 'y': [float(yj) for yj in y],
                                                                 'z': [float(zj) for zj in z]  }

    if path_json.exists():
        with open(path_json, "r", encoding='utf-8') as shock_tracking:
            stored_data = json.load(shock_tracking)

            if (type(stored_data) == dict):
                if time in stored_data:
                    print(f"Some values were already stored for this time dump {time}, they were updated.")
                    stored_data[time].update(new_data[time])
                if time not in stored_data:
                    print(f"This time_dump {time} did not have any data yet. Values were added.")
                    stored_data.update(new_data)
    else:
        print("There was no stored data. \n"
              "A '.json' containing a dict will be created. \n"
              f"This dict will only contain the data for time {time}.")
        stored_data = new_data            

    with open(path_json, "w", encoding='utf-8') as updated_shock_tracking:
        print("Writing new values")
        json.dump(stored_data, updated_shock_tracking)    


# In[8]:


updated_shock_tracking


# In[9]:


stored_data

