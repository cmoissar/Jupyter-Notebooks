#!/usr/bin/env python
# coding: utf-8

# In[1]:


#if working on lx-moissard
Cluster = 'Occ/'
run_name = '20_08_18_new_big_one_0'
filepath = '/data/Lathys/Visualisation/' + Cluster + run_name + '/ncfiles/p3_files/'

p3_file =  filepath + "p3_0000_18_08_20_t00230.nc"


# In[2]:


import netCDF4 as nc
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = [20, 15]


# In[3]:


p3_data = nc.Dataset(p3_file,'r')
print(p3_data.__dict__)


# In[ ]:


# for variable in p3_data.variables:
#     print(variable)


# In[ ]:


posx = p3_data.variables['particule_x']
posy = p3_data.variables['particule_y']
posz = p3_data.variables['particule_z']

velx = p3_data.variables['particule_vx']
vely = p3_data.variables['particule_vy']
velz = p3_data.variables['particule_vz']


# In[ ]:


plt.scatter(np.array(posx), np.array(velx), s = 0.01)


# In[ ]:


plt.scatter(np.array(posx), np.array(posy), s = 0.01)


# In[ ]:




