#!/usr/bin/env python
# coding: utf-8

# ## Import modules

# In[1]:


# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.mplot3d import Axes3D 

import pylab as pl
import numpy as np
from math import isnan

# to import modules from parent directory
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from collections.abc import Iterable
from scipy import signal
from scipy.signal import savgol_filter

import glob
import re

#import pdb
#pdb.set_trace()


# ## Vectorial Algebra

# In[2]:


# slice_ = slice(-20,20)
# slice_ = slice(slice_.start - 1, slice_.stop + 1)

# print(slice_)


# In[3]:


#All these functions use a "slice_". They are computed only where the result is needed.
#This may cause some small inaccuracies at the edges of the "slice_" for the functions using np.grad()

def curl(vector, slice_=slice(None), verbose=False):
    """
    vector = [Ax, Ay, Az]
    """
    
    if verbose:
        print("Calculating the curl of some data...")
    
    Ax, Ay, Az = vector
    
    try:
        
        extended_slice_x = slice(slice_[0].start - 1, slice_[0].stop + 1)
        extended_slice_y = slice(slice_[1].start - 1, slice_[1].stop + 1)
        extended_slice_z = slice(slice_[2].start - 1, slice_[2].stop + 1)

        extended_slice_ = (extended_slice_x, extended_slice_y, extended_slice_z)
        
        if isinstance(Ax, Iterable):
            Ax = Ax[extended_slice_]
            Ay = Ay[extended_slice_]
            Az = Az[extended_slice_]
        
        test = True
        
        if verbose:
            print("""We can take a slightly larger array for the calculations,
this lessens errors at the edges. The result is cropped back to the right size.""")
                
    except TypeError as e:
        if verbose:
            print(f"""Since {e}, we cannot take a slightly larger array for the calculations,
so the result will be slightly wrong, especially at the box's edges""")
        if isinstance(Ax, Iterable):
            Ax = Ax[slice_]
            Ay = Ay[slice_]
            Az = Az[slice_]

            test = False        
    
    def dAdx(A):
        return np.gradient(A, axis = 0) / gstep[0]
    def dAdy(A):
        return np.gradient(A, axis = 1) / gstep[1]
    def dAdz(A):
        return np.gradient(A, axis = 2) / gstep[2]
       
    Cx = dAdy(Az) - dAdz(Ay)
    Cy = dAdz(Ax) - dAdx(Az)
    Cz = dAdx(Ay) - dAdy(Ax)
    
    if test:
        Cx, Cy, Cz = Cx[1:-1, 1:-1, 1:-1], Cy[1:-1, 1:-1, 1:-1], Cz[1:-1, 1:-1, 1:-1]
    
    return [Cx, Cy, Cz]

def dot_product(vector_1, vector_2, slice_=slice(None)):
    """
    vector = [Ax, Ay, Az]
    """
    #it may be stupid to define this, np.dot(A, B) does the same thing
    
    Ax, Ay, Az = vector_1
    if isinstance(Ax, Iterable):
        Ax = Ax[slice_]
        Ay = Ay[slice_]
        Az = Az[slice_]
    Bx, By, Bz = vector_2
    if isinstance(Bx, Iterable):
        Bx = Bx[slice_]
        By = By[slice_]
        Bz = Bz[slice_]
        
    if isinstance(Ax, Iterable):
        return Ax[:]*Bx[:] + Ay[:]*By[:] + Az[:]*Bz[:]
    else:
        return Ax*Bx + Ay*By + Az*Bz
       

def cross_product(vector_1, vector_2, slice_=slice(None)):
    """
    vector = [Ax, Ay, Az]
    """
    #it may be stupid to define this, np.cross(A, B) does the same thing
    
    Ax, Ay, Az = vector_1
    if isinstance(Ax, Iterable):
        Ax = Ax[slice_]
        Ay = Ay[slice_]
        Az = Az[slice_]
    Bx, By, Bz = vector_2
    if isinstance(Bx, Iterable):
        Bx = Bx[slice_]
        By = By[slice_]
        Bz = Bz[slice_]

    Cx = Ay * Bz - Az * By
    Cy = Az * Bx - Ax * Bz
    Cz = Ax * By - Ay * Bx
    
    return [Cx, Cy, Cz]
    

def norm(vector, slice_=slice(None)):
    """
    vector = [Ax, Ay, Az]
    """
    if isinstance(vector[0], Iterable):
        return np.sqrt(dot_product(vector, vector, slice_)[:])
    else:
        return np.sqrt(dot_product(vector, vector, slice_))


# ## Physics constants & normalisation factors

# In[4]:


qe = 1.60217662e-19 #C
µ0 = 4*np.pi*1e-7
mp = 1.67262e-27 #kg
kB = 1.38064852e-23 #m².kg.s^-2.K^-1

#conversion to SI:
b = 1e-9
n = 1e6
v = 1e3
t = 11605


# ## Import Data

# ### import_data_3D

# In[5]:


def import_data_3D(filepath, date, time, str_file_type):

   # to import modules from parent directory
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
#    import pandas as pds

    from read_netcdf_3D import readNetcdfFile3D

    file_dataw  = str_file_type + '_' + date + '_t' + time + '.nc'

    dataw = {}
    
    print(f"Importing {str_file_type} 3D from {filepath}")
    readNetcdfFile3D(filepath,file_dataw, dataw)
    
#     dataw  = pds.Series(dataw)
    
    # Axes in normalised units
    dataw['x'][:] = dataw['x'][:]/dataw['c_omegapi']
    dataw['y'][:] = dataw['y'][:]/dataw['c_omegapi']
    dataw['z'][:] = dataw['z'][:]/dataw['c_omegapi']
    
    # Note: X, Y, Z that will be used for the rest of the code are therefore the ones
    #       that correspond to the last imported data
    global X, Y, Z 
    X = np.array(np.around(dataw['x'][:]))
    Y = np.array(np.around(dataw['y'][:]))
    Z = np.array(np.around(dataw['z'][:]))
    global cwp, gstep
    cwp = dataw['c_omegapi']
    gstep = dataw['gstep']
        
    return dataw


# ## Define Functions (e.g. Current J, ...)

# In[6]:


def identity(data, slice_=slice(None)):
    '''
    returns a slice of data
    '''
    if len(data) == 3:
        Ax, Ay, Az = data
        return Ax[slice_], Ay[slice_], Az[slice_]
    else:        
        return data[slice_]

def J(B, slice_=slice(None)):
    '''
    Computes the current on a slice of the box
    This is way more memory efficient than computing the current on the whole box
    then slicing.
    The one problem with it though, is wall effects. np.gradient does interpolations at
    the edges of the box to keep the same shape from A to np.grad(A)
    '''
    j = 1./(µ0*cwp*1000) # unit of J: (nA/m²)
    Jx, Jy, Jz = j * curl(B, slice_)
    return Jx, Jy, Jz

def Jx(B, slice_=slice(None)):
    return J(B, slice_)[0]
def Jy(B, slice_=slice(None)):
    return J(B, slice_)[1]
def Jz(B, slice_=slice(None)):
    return J(B, slice_)[2]

def aplatir(conteneurs):
    '''
    returns a float from a 0D array, or a list from a np.array. Useful sometimes.
    '''
    result = [item for conteneur in conteneurs for item in conteneur]
    if len(result) == 1:
        return result[0]
    else:
        return result


# In[7]:


def convert_coord_to_indices( coord ):
    indice_x = aplatir(np.where(abs(X-coord[0])==min(abs(X-coord[0]))))
    indice_y = aplatir(np.where(abs(Y-coord[1])==min(abs(Y-coord[1]))))
    indice_z = aplatir(np.where(abs(Z-coord[2])==min(abs(Z-coord[2]))))
    
    indices = [indice_x, indice_y, indice_z]

    for i in range(0, len(indices)):
        if isinstance(indices[i], Iterable):
            if len(indices[i])>1:
                indices[i] = indices[i][0]        
    
    return indices   

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# In[ ]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# ## Diagnostics

# ### find_bow_shock_and_magnetopause(...) and compute_global_geometry(...)

# In[8]:


from IPython.core.debugger import set_trace

def find_bow_shock_and_magnetopause(str_coord, B, N, V, loc=None):
    '''
    the option "loc" makes it possible to find the bow shock at another position than the
    usual nose, yup, ydown, zup, zdown
    '''
    
    global nx0, ny0, nz0
    nx,  ny,  nz  = len(X), len(Y), len(Z)
    # Location of the planet is defined in the .ncfiles as (x,y,z) = (0,0,0)
    try:
        nx0, ny0, nz0 = ( int(np.where(abs(X)==min(abs(X)))[0][0]),
                          int(np.where(abs(Y)==min(abs(Y)))[0][0]), 
                          int(np.where(abs(Z)==min(abs(Z)))[0][0])  )
    except SyntaxError as e:
        nx0, ny0, nz0 = ( int(np.where(abs(X)==min(abs(X)))[0]),
                          int(np.where(abs(Y)==min(abs(Y)))[0]), 
                          int(np.where(abs(Z)==min(abs(Z)))[0])  )

#     print("nx0", nx0, "ny0", ny0, "nz0", nz0)

    #These two numbers were found empirically to yield best results.
    #if gstep is different than 1, They should probably be modified.
    global dL
    dL = 5 #dL is the thickness of slices in units of di
    global dl 
    gst = int(np.mean(gstep))
    dl = 5 #dL // gst #TODO: for consistency, I should define dlx, dly, dlz

    Bx, By, Bz = tuple(B)
    Vx, Vy, Vz = tuple(V)
    
    N = N
    
    if loc:
        sx = aplatir(np.where(abs(X-loc[0])==min(abs(X-loc[0]))))
        sy = aplatir(np.where(abs(Y-loc[1])==min(abs(Y-loc[1]))))
        sz = aplatir(np.where(abs(Z-loc[2])==min(abs(Z-loc[2]))))
        
        shift = [sx, sy, sz]
        
        for i in range(0, 3):
            if isinstance(shift[i], Iterable):
                if len(shift[i])>1:
                    shift[i] = shift[i][0]        
                 

        shift = (shift[0] - nx0,
                 shift[1] - ny0,
                 shift[2] - nz0)
    else: 
        shift = (0, 0, 0)

    sx, sy, sz = shift
    
    nxs = nx0 + shift[0]
    nys = ny0 + shift[1]
    nzs = nz0 + shift[2]
        
    if (str_coord=='X'):
        coord = X
        slice_x = slice(None)
        slice_y = slice(nys-dl, nys+dl)
        slice_z = slice(nzs-dl, nzs+dl)
        slices = (slice_x, slice_y, slice_z)

        #Values for x are higher, because the shock takes the solar wind head-on
        #Dividing b_slice by 2 allows to use the same test for all coords.
        b_slice = np.sqrt( Bx[slices]**2
                          +By[slices]**2
                          +Bz[slices]**2 ).mean(axis=(1, 2))
        j_slice = np.sqrt(sum([ji**2 for ji in J(B, slices)])).mean(axis=(1, 2))/2
        jy_slice = abs(Jy(B, slices)).mean(axis=(1, 2))
        jz_slice = abs(Jz(B, slices)).mean(axis=(1, 2))
        jyz_slice = np.sqrt(jy_slice**2 + jz_slice**2)
        v_slice = np.sqrt( Vx[slices]**2
                          +Vy[slices]**2
                          +Vz[slices]**2 ).mean(axis=(1, 2))/2
        n_slice = N[slices].mean(axis=(1, 2))

    if (str_coord=='Y'):
        coord = Y
        gstep_coord = abs(np.mean(coord[1:] - coord[:-1]))
        slice_x = slice(nxs-dl, nxs+dl)
        slice_y = slice(None)
        slice_z = slice(nzs-dl, nzs+dl)
        slices = (slice_x, slice_y, slice_z)

        b_slice = np.sqrt( Bx[slices]**2
                          +By[slices]**2
                          +Bz[slices]**2 ).mean(axis=(0, 2))
        j_slice = np.sqrt(sum([ji**2 for ji in J(B, slices)])).mean(axis=(0, 2))
        jy_slice = abs(Jy(B, slices)).mean(axis=(0, 2))
        jz_slice = abs(Jz(B, slices)).mean(axis=(0, 2))
        jyz_slice = np.sqrt(jy_slice**2 + jz_slice**2)
        v_slice = np.sqrt( Vx[slices]**2
                          +Vy[slices]**2
                          +Vz[slices]**2 ).mean(axis=(0, 2))
        n_slice = N[slices].mean(axis=(0, 2))

    if (str_coord=='Z'):
        coord = Z
        gstep_coord = abs(np.mean(coord[1:] - coord[:-1]))
        slice_x = slice(nxs-dl, nxs+dl)
        slice_y = slice(nys-dl, nys+dl)
        slice_z = slice(None)
        slices = (slice_x, slice_y, slice_z)

        b_slice = np.sqrt( Bx[slices]**2
                          +By[slices]**2
                          +Bz[slices]**2 ).mean(axis=(0, 1))
        j_slice = np.sqrt(sum([ji**2 for ji in J(B, slices)])).mean(axis=(0, 1))
        jy_slice = abs(Jy(B, slices)).mean(axis=(0, 1))
        jz_slice = abs(Jz(B, slices)).mean(axis=(0, 1))
        jyz_slice = np.sqrt(jy_slice**2 + jz_slice**2)
        v_slice = np.sqrt( Vx[slices]**2
                          +Vy[slices]**2
                          +Vz[slices]**2 ).mean(axis=(0, 1))
        n_slice = N[slices].mean(axis=(0, 1))

    n_slice = savgol_filter(n_slice, 51, 3)
   
    #find bow shock using a local max of current J    
    def find_bow_shock():
        
        '''
        This function finds the bow shock in two steps:
        1) Find regions where it would be reasonable to find the bow shock:
           -the magnetic field is quite high
           -the region is not too far from the planet
           -there are gradients of velocity (and magnetic field)
        2) Within these regions, look for a local maximum of the current     
        '''

        #general tests for magnetosheath:
        test_j_large   = j_slice > 1.7*np.median(j_slice)
        test_close_to_planet = (abs(coord)<min((1./2)*len(coord)*np.mean(coord[1:]-coord[:-1])
                                                       for coord in [X,Y,Z])-15)

        test_coord_up   = (coord > 0)
        test_coord_down = (coord < 0)

        test_up =    (  test_j_large
                      & test_close_to_planet
                      & test_coord_up        )


        maximums = signal.argrelextrema(j_slice, np.greater, order=1+int(5/np.mean(gstep.data)))

        #This is need to discrimitate between
        #the bow shock and the interplanetary shock
        if str_coord=='X':
            test_b_grad_up = (np.gradient(b_slice) < -0.5*np.mean(gstep.data) )
            test_up = test_up & test_b_grad_up

        test_down =  (  test_j_large
                      & test_close_to_planet
                      & test_coord_down      )

        #DOUBLE LOOP. THIS IS TERRIBLY INEFFICIENT AND UGLY
        def def_coord_bow_shock(test, loc='down'):

            where_test = aplatir(np.where(test))

            if loc=='up':
                where_test.reverse()

            for t in where_test:
                for local_max in j_slice[maximums]:
        #             print(f"Comparing {j_slice[t]} with {local_max}, which yields {j_slice[t] == local_max}")
        #             import time
        #             time.sleep(0.5)
                    if j_slice[t] == local_max:
                        return coord[t]
            
            print("Had some trouble finding coord_bow_shock_down. Returned 0")
            return 0

        coord_bow_shock_up   = def_coord_bow_shock(test_up, 'up')    
        coord_bow_shock_down = def_coord_bow_shock(test_down)


        return coord_bow_shock_up, coord_bow_shock_down

    coord_bow_shock_up, coord_bow_shock_down = find_bow_shock()   
    
    #indeed, it might not make sense to look for the magnetopause at some places
    #this should be adressed more beautifuly for completeness
    if loc: 
        return coord_bow_shock_up, coord_bow_shock_down, np.nan, np.nan

    # find magnetopause using large variation of n and local max of current Jz    
    def find_magnetopause(str_coord):
        
        '''
        This function looks for strong gradients of density.
        The magnetopause is defined as the local max of Jz in these density gradients.
        '''

        test_planet = (15 < abs(coord)) & (abs(coord) < 80)
        test_coord_up  = (coord > 0)
        test_coord_down  = (coord < 0)
        test_grad_n_up   = (np.gradient(n_slice) > 0.2*max(np.gradient(n_slice)[test_planet])) & test_coord_up
        test_grad_n_down = (np.gradient(n_slice) < 0.2*min(np.gradient(n_slice)[test_planet])) & test_coord_down

        test_up   = test_grad_n_up & test_coord_up & test_planet
        test_down = test_grad_n_down & test_coord_down & test_planet

        # def give_center_of_multiple_ones(test): 
        #     count  = 0 
        #     counts = [] 
        #     for t in test: 
        #         if t: 
        #             count += 1 
        #         else:    
        #             count = 0 
        #         counts.append(count) 
        #         end = counts.index(max(counts)) 
        #         start = end - max(counts) + 1
        #         center = int(start + (end-start)/2) 
        #     return center

        def intersection(lst1, lst2): 
            lst3 = [value for value in lst1 if value in lst2] 
            return lst3 

        maximums = signal.argrelextrema(jyz_slice, np.greater, order=4)

        #set_trace()

        if str_coord=='X':
            jyz_max_local_max_up = max(intersection(jyz_slice[maximums], jyz_slice[test_up]))
            i_m_up = aplatir(np.where(jyz_slice == jyz_max_local_max_up))
            coord_magnetopause_up = coord[i_m_up]

            coord_magnetopause_down = 0
        elif (str_coord == 'Y'):
            jyz_max_local_max_up = second_largest(intersection(jyz_slice[maximums], jyz_slice[test_up]))
            i_m_up = aplatir(np.where(jyz_slice == jyz_max_local_max_up))
            coord_magnetopause_up = coord[i_m_up]

            jyz_max_local_max_down = second_largest(intersection(jyz_slice[maximums], jyz_slice[test_down]))
            i_m_down = aplatir(np.where(jyz_slice == jyz_max_local_max_down))
            coord_magnetopause_down = coord[i_m_down]
        elif (str_coord == 'Z'):
            jyz_max_local_max_up = max(intersection(jyz_slice[maximums], jyz_slice[test_up]))
            i_m_up = aplatir(np.where(jyz_slice == jyz_max_local_max_up))
            coord_magnetopause_up = coord[i_m_up]
           
            jyz_max_local_max_down = max(intersection(jyz_slice[maximums], jyz_slice[test_down]))
            i_m_down = aplatir(np.where(jyz_slice == jyz_max_local_max_down))
            coord_magnetopause_down = coord[i_m_down]


        print(coord_magnetopause_up)
        print(coord_magnetopause_down)
        
        return coord_magnetopause_up, coord_magnetopause_down

    coord_magnetopause_up, coord_magnetopause_down = find_magnetopause(str_coord)       
    
    return coord_bow_shock_up, coord_bow_shock_down, coord_magnetopause_up, coord_magnetopause_down

### If need to check what is going on in find_bow_shock()

# plt.close('all')
# plt.plot(coord, b_slice/50, label='b slice')
# plt.plot(coord, np.gradient(b_slice), label='gradient')
# plt.plot(coord, test_up, label='test up')
# plt.plot(coord, test_down, label='test down')
# plt.ylim([-5,5])
# plt.xlim([50,750])
# plt.legend()
# plt.show()

# plt.plot(coord, n_slice/10, label='n slice')
# plt.plot(coord, np.gradient(n_slice), label='gradient')
# plt.plot(coord, test_up, label='test up')
# plt.plot(coord, test_down, label='test down')
# plt.ylim([-5,5])
# plt.xlim([50,750])
# plt.legend()
# plt.show()


# In[9]:


def compute_global_geometry(B, N, V, metadata, time):
    
    global x_bow_shock, x_magnetopause, y_bow_shock_up, y_bow_shock_down, y_magnetopause_up, y_magnetopause_down, z_bow_shock_up, z_bow_shock_down, z_magnetopause_up, z_magnetopause_down
    x_bow_shock   , _               , x_magnetopause   , _                   = find_bow_shock_and_magnetopause('X', B, N, V)
    y_bow_shock_up, y_bow_shock_down, y_magnetopause_up, y_magnetopause_down = find_bow_shock_and_magnetopause('Y', B, N, V)
    z_bow_shock_up, z_bow_shock_down, z_magnetopause_up, z_magnetopause_down = find_bow_shock_and_magnetopause('Z', B, N, V)
    
    #I don't think that this is recommanded practice.
    #global variables should be avoided. But well. It's the end of my phd, time is not with me.
    global x_is
    x_is = find_ip_shock(V, metadata, time)
    
    global x_le
    x_le = find_mc_leading_edge(B, N, metadata, time)
    
    return x_bow_shock, x_magnetopause, y_bow_shock_up, y_bow_shock_down, y_magnetopause_up, y_magnetopause_down, z_bow_shock_up, z_bow_shock_down, z_magnetopause_up, z_magnetopause_down 


# ### find_ip_shock(V, ...) and find_mc_leading_edge(B, N, ...)

# In[10]:


def find_ip_shock(V, metadata, time):
    
    #metadata = {'t_shock_entrance' : 130,
    #            't_shock_exit'     : 220,
    #            't_MC_entrance'    : 130,
    #            't_MC_exit'        : 300}
    
    time = int(time[1:])
    
    if (time < metadata['t_shock_entrance'] or time > metadata['t_shock_exit']):
        return np.array([np.nan]) #This is the same formatting than when there is a result
    
    Vx, Vy, Vz = tuple(V)    

    slice_x = slice(None)
    #looking somewhere as far away as possible from the magnetosheath
    #the magnetosheath would complicate the detection
    slice_y = slice(dl, 3*dl)
    slice_z = slice(dl, 3*dl)
    slices = (slice_x, slice_y, slice_z)

    v_slice = np.sqrt( Vx[slices]**2
                      +Vy[slices]**2
                      +Vz[slices]**2 ).mean(axis=(1, 2))/2
    grad_v = np.gradient(v_slice)
# note: absurd values can easily be removed in Story_Reader anyway
#     test_non_absurd = abs(X < 500)
#     if not(len(grad_v == np.nanmax(grad_v[test_non_absurd]))):
#         return np.nan
    ix_is = np.where(grad_v == np.nanmax(grad_v))
    x_is = X[ix_is]    

    return x_is

def find_mc_leading_edge(B, N, metadata, time):
    
    #metadata = {'t_shock_entrance' : 130,
    #            't_shock_exit'     : 220,
    #            't_MC_entrance'    : 130,
    #            't_MC_exit'        : 300}
    
    time = int(time[1:])
    
    if (time < metadata['t_MC_entrance'] or time > metadata['t_MC_exit']):
        return np.nan
    
    Bx, By, Bz = tuple(B)   

    slice_x = slice(None)
    slice_y = slice(dl, 3*dl)
    slice_z = slice(dl, 3*dl)
    slices = (slice_x, slice_y, slice_z)

    #smoothing the data avoids false positive when there are a lot of fluctuations
    from scipy.signal import savgol_filter

    b_slice = np.sqrt( Bx[slices]**2
                      +By[slices]**2
                      +Bz[slices]**2 ).mean(axis=(1, 2))/2
    b_slice = savgol_filter(b_slice, 51, 3)
    j_slice = np.sqrt(sum([ji**2 for ji in J(B, slices)])).mean(axis=(1, 2))
    n_slice = N[slices].mean(axis=(1, 2))
    n_slice = savgol_filter(n_slice, 51, 3)

    grad_n = np.gradient(n_slice)
    grad_b = np.gradient(b_slice)
    test_non_absurd = (abs(X) < 800) 
    if not(isnan(x_is)):
        test_non_absurd = test_non_absurd & (abs(X - x_is) > 45)


    test_grad_n = grad_n < -1*np.nanmean(abs(grad_n))
    test_grad_b = grad_b > 1*np.nanmean(abs(grad_b))
    test_le = test_grad_n & test_grad_b & test_non_absurd 
    
    ix_le = np.where(j_slice == np.nanmax(j_slice[np.where(test_le)]))

    x_le = X[ix_le]

    return x_le


# ### construct_box(...) and plot_boxes(...)

# In[11]:


def construct_box(str_coord, coord_bow_shock, coord_magnetopause):    
    
    global edge, size_cubes
    
    gst = int(np.mean(gstep))
    edge = max(gst, (4 // gst) * gst) #gives a reasonably large edge, as a multiple of gstep
    size_cubes = abs(x_bow_shock-x_magnetopause) - 2*edge #TODO: try other values
        
    magnetosheath_half_width = abs(coord_bow_shock-coord_magnetopause)/2
    
    if str_coord=='upstream':
        x_max = coord_bow_shock + 2*magnetosheath_half_width + size_cubes/2
        x_min = coord_bow_shock + 2*magnetosheath_half_width - size_cubes/2
        y_max =  size_cubes/2
        y_min = -size_cubes/2
        z_max =  size_cubes/2
        z_min = -size_cubes/2
    
    if str_coord=='nose':
        x_max = coord_magnetopause + magnetosheath_half_width + size_cubes/2
        x_min = coord_magnetopause + magnetosheath_half_width - size_cubes/2
        y_max =  size_cubes/2
        y_min = -size_cubes/2
        z_max =  size_cubes/2
        z_min = -size_cubes/2
    
    if str_coord=='yup':
        x_max =  size_cubes/2
        x_min = -size_cubes/2
        y_max = coord_magnetopause + magnetosheath_half_width + size_cubes/2
        y_min = coord_magnetopause + magnetosheath_half_width - size_cubes/2
        z_max =  size_cubes/2
        z_min = -size_cubes/2
        
    if str_coord=='ydown':
        x_max =  size_cubes/2
        x_min = -size_cubes/2
        y_max = coord_magnetopause - magnetosheath_half_width + size_cubes/2
        y_min = coord_magnetopause - magnetosheath_half_width - size_cubes/2
        z_max =  size_cubes/2
        z_min = -size_cubes/2
          
    if str_coord=='zup':
        x_max =  size_cubes/2
        x_min = -size_cubes/2
        y_max =  size_cubes/2
        y_min = -size_cubes/2
        z_max =  coord_magnetopause + magnetosheath_half_width + size_cubes/2
        z_min =  coord_magnetopause + magnetosheath_half_width - size_cubes/2
        
    if str_coord=='zdown':
        x_max =  size_cubes/2
        x_min = -size_cubes/2
        y_max =  size_cubes/2
        y_min = -size_cubes/2
        z_max =  coord_magnetopause - magnetosheath_half_width + size_cubes/2
        z_min =  coord_magnetopause - magnetosheath_half_width - size_cubes/2
    
    return (x_max,x_min,y_max,y_min,z_max,z_min)

def construct_box_indexes(str_coord, coord_bow_shock, coord_magnetopause):
    
    (x_max,x_min,y_max,y_min,z_max,z_min) = construct_box(str_coord, coord_bow_shock, coord_magnetopause)
    
    ix_min = int(np.where(x_min<=X)[0][0])    
    ix_max = int(np.where(x_max<=X)[0][0])
    iy_min = int(np.where(y_min<=Y)[0][0])
    iy_max = int(np.where(y_max<=Y)[0][0])
    iz_min = int(np.where(z_min<=Z)[0][0])
    iz_max = int(np.where(z_max<=Z)[0][0])
    
    return (ix_max,ix_min,iy_max,iy_min,iz_max,iz_min)    


# In[12]:


def make_boxes_JSON_serializable(boxes):
    
    '''
    .json files are not happy with numbers with a type = np.float32
    a.item() converts a from np.float32 to float
    '''
   
    import numbers

    for box in boxes:
        for element in boxes[box]:
            if (type(boxes[box][element])==np.float32):
                boxes[box][element] = boxes[box][element].item()
            else:
                for sub_element in boxes[box][element]:   
                    if isinstance(sub_element, numbers.Number):
                        sub_element = sub_element.item()
                    elif ( type(boxes[box][element][sub_element])==np.float32 ):
                        boxes[box][element][sub_element] = boxes[box][element][sub_element].item()
                        
    return boxes

def create_boxes_dictionary():

    boxes = { 'upstream' : {"coord_bow_shock": x_bow_shock      , 'coord_magnetopause': x_magnetopause      } ,
              'nose'     : {"coord_bow_shock": x_bow_shock      , 'coord_magnetopause': x_magnetopause      } ,
              'yup'      : {"coord_bow_shock": y_bow_shock_up   , 'coord_magnetopause': y_magnetopause_up   } ,
              'ydown'    : {"coord_bow_shock": y_bow_shock_down , 'coord_magnetopause': y_magnetopause_down } ,
              'zup'      : {"coord_bow_shock": z_bow_shock_up   , 'coord_magnetopause': z_magnetopause_up   } ,
              'zdown'    : {"coord_bow_shock": z_bow_shock_down , 'coord_magnetopause': z_magnetopause_down }   } 

    for box in boxes:             

        xmax,xmin,ymax,ymin,zmax,zmin = construct_box(box, boxes[box]["coord_bow_shock"],
                                                           boxes[box]["coord_magnetopause"])

        boxes[box].update( {"box_indexes": { 'xmax': xmax,
                                             'xmin': xmin, 
                                             'ymax': ymax,
                                             'ymin': ymin, 
                                             'zmax': zmax,
                                             'zmin': zmin   } } )

        center = ( xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2, zmin + (zmax - zmin)/2 )

        boxes[box].update( { "center": center } )
    
    return boxes


# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_boxes():
    
    boxes = create_boxes_dictionary()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ones = np.ones(4).reshape(2, 2)

    def x_y_edge(x_range, y_range, z_range):
        xx, yy = np.meshgrid(x_range, y_range)

        for value in [0, 1]:
            ax.plot_wireframe(xx, yy, z_range[value]*ones, color="r")
            ax.plot_surface(xx, yy, z_range[value]*ones, color="r", alpha=0.2)


    def y_z_edge(x_range, y_range, z_range):
        yy, zz = np.meshgrid(y_range, z_range)

        for value in [0, 1]:
            ax.plot_wireframe(x_range[value]*ones, yy, zz, color="r")
            ax.plot_surface(x_range[value]*ones, yy, zz, color="r", alpha=0.2)


    def x_z_edge(x_range, y_range, z_range):
        xx, zz = np.meshgrid(x_range, z_range)

        for value in [0, 1]:
            ax.plot_wireframe(xx, y_range[value]*ones, zz, color="r")
            ax.plot_surface(xx, y_range[value]*ones, zz, color="r", alpha=0.2)


    def rect_prism(x_range, y_range, z_range):
        x_y_edge(x_range, y_range, z_range)
        y_z_edge(x_range, y_range, z_range)
        x_z_edge(x_range, y_range, z_range)
        
    for box in boxes:             

        xmax,xmin,ymax,ymin,zmax,zmin = construct_box(box, boxes[box]["coord_bow_shock"],
                                                           boxes[box]["coord_magnetopause"])
        
        rect_prism(np.array([xmin, xmax]),
                   np.array([ymin, ymax]),
                   np.array([zmin, zmax]))
  
 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #plot box centers
    for box in boxes:
        print(f"box {box} has its center at {boxes[box]['center']}")
        
        ax.scatter(boxes[box]['center'][0],
                   boxes[box]['center'][1],
                   boxes[box]['center'][2])

    plt.show()


# ### calculate_bow_shock_parameters(...)

# In[14]:


def bow_shock_normale(loc, B, N, V, along='X', d=12):
    
    '''
    Calculates the normal of the shock upstream of the coordinates 'loc'
    '''
                
    #Todo: along='flow'
        
    if (along == 'X'):
        loc_yu = (loc[0], loc[1]+d, loc[2])
        loc_yd = (loc[0], loc[1]-d, loc[2])
                
        x_bow_shock_up_yu, _, _, _ = find_bow_shock_and_magnetopause('X', B, N, V, loc_yu)
        x_bow_shock_up_yd, _, _, _ = find_bow_shock_and_magnetopause('X', B, N, V, loc_yd)
        
        uxy = (x_bow_shock_up_yu - x_bow_shock_up_yd, loc_yu[1] - loc_yd[1], 0)
        
        loc_zu = (loc[0], loc[1], loc[2]+d)
        loc_zd = (loc[0], loc[1], loc[2]-d)
        x_bow_shock_up_zu, _, _, _ = find_bow_shock_and_magnetopause('X', B, N, V, loc_zu)
        x_bow_shock_up_zd, _, _, _ = find_bow_shock_and_magnetopause('X', B, N, V, loc_zd)
        
        uxz = (x_bow_shock_up_zu - x_bow_shock_up_zd, 0, loc_zu[2] - loc_zd[2])

        vector = cross_product(uxy, uxz)
        vector = vector / norm(vector)
                
        origin_x, _, _, _ = find_bow_shock_and_magnetopause('X', B, N, V, loc)
        origin_y = loc[1]
        origin_z = loc[2]
        
        origin = (origin_x, origin_y, origin_z)
        
        return origin, vector


# In[15]:


def calculate_bow_shock_parameters(loc, B, N, V, T):    
     
    gamma = 5./3
    origin, vector = bow_shock_normale(loc, B, N, V, along='X', d=12)
    magnetosheath_half_width = abs(x_bow_shock-x_magnetopause)/2

    loc_upstream = (origin[0]+2*magnetosheath_half_width, origin[1], origin[2])
    xmin, ymin, zmin = loc_upstream[0]-size_cubes/2, loc_upstream[1]-size_cubes/2, loc_upstream[2]-size_cubes/2
    xmax, ymax, zmax = loc_upstream[0]+size_cubes/2, loc_upstream[1]+size_cubes/2, loc_upstream[2]+size_cubes/2
    ixmin, iymin, izmin = convert_coord_to_indices((xmin, ymin, zmin))
    ixmax, iymax, izmax = convert_coord_to_indices((xmax, ymax, zmax))

    slice_upstream = (slice(ixmin, ixmax), slice(iymin, iymax), slice(izmin, izmax)) 
        
    B_loc_upstream = identity(B, slice_upstream)
    B_loc_upstream = (np.mean(B_loc_upstream[0]), np.mean(B_loc_upstream[1]), np.mean(B_loc_upstream[2]))
        
    V_loc_upstream = identity(V, slice_upstream)
    V_loc_upstream = (np.mean(V_loc_upstream[0]), np.mean(V_loc_upstream[1]), np.mean(V_loc_upstream[2]))
    
    N_loc_upstream = identity(N, slice_upstream)
    N_loc_upstream = np.mean(N_loc_upstream)
    
    T_loc_upstream = identity(T, slice_upstream)
    T_loc_upstream = np.mean(T_loc_upstream)
    
    V_A = norm(B_loc_upstream)*b / np.sqrt(µ0 * mp * N_loc_upstream * n)
    M_A = norm(V_loc_upstream)*v / V_A
 
    C_S = np.sqrt(gamma * kB * T_loc_upstream*t / mp)
    M_S = norm(V_loc_upstream)*v / C_S
 
    
    theta = dot_product(vector, B_loc_upstream )
    theta = np.arccos( theta / (norm(vector) * norm(B_loc_upstream)) ) * (180/np.pi)
    if (theta > 90):
        theta = 180 - theta
    
    V_MS = np.sqrt( (1./2)* (  C_S**2 + V_A**2 
                             + np.sqrt( (C_S**2 + V_A**2)**2
                                       - 4*C_S**2 * V_A**2 * np.cos(theta*np.pi/180 )**2 ) ) )
    M_MS = norm(V_loc_upstream)*v / V_MS
        
    beta = kB*(n*N_loc_upstream)*(t*T_loc_upstream)*(2*µ0/(b*norm(B_loc_upstream))**2)
    
    return {'Alfven Mach number': M_A, 'theta_Bn': theta, 'Beta': beta,
            'sonic Mach number': M_S,
            'fast magnetosonic Mach number': M_MS}
    


# ### compute_xxx_in_cubes(...)

# #### Compute data and rms

# In[16]:


def compute_data_in_cubes(data1, data2=None, function1=identity, function_both=None):
    '''
    Returns the mean value of data in the different cubes.
    If data is a vector (len(data)==3), then the function returns the mean value the norm(data[cubes])
    '''
    
    boxes = create_boxes_dictionary()
    
    data_in_boxes = []
    
    for box in boxes:     
                        
        ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes(box, boxes[box]["coord_bow_shock"],
                                                                         boxes[box]["coord_magnetopause"])
        
        slices = (slice(ixmin, ixmax), slice(iymin, iymax), slice(izmin, izmax))
        
        if (not(data2)):
                        
            result = function1( data1, slices )
            
        else:
            result = function_both( function1( data1, slices ), [data2[0][slices],
                                                                 data2[1][slices],
                                                                 data2[2][slices]] )
            
        if (len(result) == 3):
                result = np.sqrt( result[0]**2 + result[1]**2 + result[2]**2 )
                  
        data_in_boxes.append(np.mean(result))

    return data_in_boxes
    

def compute_RMS_in_cubes(data):
        
    def compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, avg):        
        avg_array = np.ones([ixmax-ixmin,iymax-iymin,izmax-izmin]) * avg        
        rms = np.sqrt(np.mean( (data[ixmin:ixmax,iymin:iymax,izmin:izmax]-avg_array)**2 ))
        return rms        
    
    tests = {}
    
    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('upstream', x_bow_shock, x_magnetopause)
    data_upstream = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_upstream = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)
    
    tests.update({'upstream': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})

    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('nose', x_bow_shock, x_magnetopause)
    data_nose = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_nose = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)

    tests.update({'nose': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})
    
    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('yup', y_bow_shock_up, y_magnetopause_up)
    data_yup = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_yup = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)   
    
    tests.update({'yup': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})

    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('ydown', y_bow_shock_down, y_magnetopause_down)
    data_ydown = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_ydown = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)
    
    tests.update({'ydown': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})
               
    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('zup', z_bow_shock_up, z_magnetopause_up)
    data_zup = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_zup = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)
    
    tests.update({'zup': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})
              
    ixmax,ixmin,iymax,iymin,izmax,izmin = construct_box_indexes('zdown', z_bow_shock_down, z_magnetopause_down)
    data_zdown = np.mean(data[ixmin:ixmax,iymin:iymax,izmin:izmax])
    RMS_zdown = compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, data_upstream)
    
    tests.update({'zdown': data[ixmin:ixmax,iymin:iymax,izmin:izmax]})

    return RMS_upstream, RMS_nose, RMS_yup, RMS_ydown, RMS_zup, RMS_zdown, tests


# In[ ]:





# #### Test compute_RMS (...)

# In[17]:


import numpy as np

print('testing compute_RMS(...):')

data = np.random.randn(10, 10, 10)

def compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, avg):        
    avg_array = np.ones([ixmax-ixmin,iymax-iymin,izmax-izmin]) * avg        
    rms = np.sqrt(np.mean( (data[ixmin:ixmax,iymin:iymax,izmin:izmax]-avg_array)**2 ))
    return rms    

#should give around 1
print(f"This should be close to 1: {compute_RMS(10,0,10,0,10,0, 0)}")

data = np.ones([10, 10, 10])

def compute_RMS(ixmax,ixmin,iymax,iymin,izmax,izmin, avg):        
    avg_array = np.ones([ixmax-ixmin,iymax-iymin,izmax-izmin]) * avg        
    rms = np.sqrt(np.mean( (data[ixmin:ixmax,iymin:iymax,izmin:izmax]-avg_array)**2 ))
    return rms    

#should give 0
print(f"This should be close to 0: {compute_RMS(10,0,10,0,10,0, 1)}")


# #### IndexTracker (doesn't seem to work when defined here. Works well directly in the working.ipynb)

# In[18]:


class IndexTracker(object):

#     import pdb; pdb.set_trace()
    
    global fontsize
    fontsize = 16

    def __init__(self, ax, X, plane):
        global plan
        plan = plane

        self.ax = ax
        self.X = X

        if plan=='xy':
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2
            self.im = ax.imshow(self.X[:, :, self.ind])
        if plan=='xz':
            rows, self.slices, cols = X.shape
            self.ind = self.slices//2
            self.im = ax.imshow(self.X[:, self.ind, :])
        if plan=='yz':
            self.slices, rows, cols = X.shape
            self.ind = self.slices//2
            self.im = ax.imshow(self.X[self.ind, :, :])

        min_value = int(np.min(X))
        max_value = int(np.median(X[np.isfinite(X)])*5)   
        # Number of color levels
        levels = MaxNLocator(nbins=255).tick_values(min_value, max_value)
        nb_ticks = 10
        cbar_ticks = MaxNLocator(nbins=nb_ticks).tick_values(min_value, max_value)
        cbar_ticks = ['{:.0f}'.format(tick) for tick in cbar_ticks]
        while ( len(cbar_ticks) <= nb_ticks ) :
            cbar_ticks.append(r"$\infty$")
        cmap = plt.get_cmap('plasma')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = self.im.axes.figure.colorbar(self.im, cax=cax, cmap=cmap, norm=norm)
        cbar.ax.set_yticklabels(cbar_ticks) #, fontsize=16, weight='bold')

        self.update(ax)

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update(ax)

    def update(self, ax):
        if plan=='xy':
            self.im.set_data(self.X[:, :, self.ind])
            ax.set_title(f'''Use scroll wheel to navigate images. Display of plane ({plan}).
z = {self.ind}''')
            ax.set_ylabel('x', weight='bold', fontsize=fontsize)
            ax.set_xlabel('y', weight='bold', fontsize=fontsize)
        if plan=='xz':
            self.im.set_data(self.X[:, self.ind, :])
            ax.set_title(f'''Use scroll wheel to navigate images. Display of plane ({plan}).
y = {self.ind}''')
            ax.set_ylabel('x', weight='bold', fontsize=fontsize)
            ax.set_xlabel('z', weight='bold', fontsize=fontsize)
        if plan=='yz':
            self.im.set_data(self.X[self.ind, :, :])
            ax.set_title(f'''Use scroll wheel to navigate images. Display of plane ({plan}).
x = {self.ind}''')
            ax.set_ylabel('y', weight='bold', fontsize=fontsize)
            ax.set_xlabel('z', weight='bold', fontsize=fontsize)
        self.im.axes.figure.canvas.draw()


# ### Plot colormaps

# In[19]:


# Define a colormap plot function
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
def plot_colormap(A, title, label, plane,
                  ratio_max_to_med = 4,
                  with_dots = False, loop = False,
                  normales = None,
                  save_dir = None, t_label = None, 
                  zoom = None, density = 1, linewidth = 1,
                  streamplot = None, Bx = None, Bj = None):
     
    min_value = int(np.min(A))
    max_value = int(np.median(A[np.isfinite(A)])*ratio_max_to_med)   
    # Number of color levels
    levels = MaxNLocator(nbins=255).tick_values(min_value, max_value)
    nb_ticks = 10
    cbar_ticks = MaxNLocator(nbins=nb_ticks).tick_values(min_value, max_value)
    cbar_ticks = ['{:.0f}'.format(tick) for tick in cbar_ticks]
    while ( len(cbar_ticks) <= nb_ticks ) :
        cbar_ticks.append(r"$\infty$")
    cmap = plt.get_cmap('plasma')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    if with_dots:
        x_dots = [x_bow_shock, x_magnetopause, 0             , 0               , 0                , 0                  ]
        y_dots = [0          , 0             , y_bow_shock_up, y_bow_shock_down, y_magnetopause_up, y_magnetopause_down]
        z_dots = [0          , 0             , z_bow_shock_up, z_bow_shock_down, z_magnetopause_up, z_magnetopause_down]    
    
    plt.close('all')
             
    if zoom:
        xmin, xmax, ymin, ymax, zmin, zmax = zoom

    else:
        xmin, xmax, ymin, ymax, zmin, zmax = (min(X), max(X),
                                              min(Y), max(Y),
                                              min(Z), max(Z))
    
    if plane=='xy':
        plot = pl.pcolor(X, Y, A.T, cmap=cmap, norm=norm) 
        plt.xlabel('x', fontsize = 16, weight="bold")
        plt.ylabel('y', fontsize = 16, weight="bold")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        if with_dots:
            plt.plot([x_dots], [y_dots], marker='o', markersize=3, color="red")
            plt.plot(x_is, ymax - 20, marker='x', markersize=6, color='cyan')
            plt.plot(x_is, 0        , marker='x', markersize=6, color='cyan')
            plt.plot(x_is, ymin + 20, marker='x', markersize=6, color='cyan')
            plt.plot(x_le, ymax - 20, marker='+', markersize=6, color='cyan')
            plt.plot(x_le, 0        , marker='+', markersize=6, color='cyan')
            plt.plot(x_le, ymin + 20, marker='+', markersize=6, color='cyan')
    if plane=='xz':
        plot = pl.pcolor(X, Z, A.T, cmap=cmap, norm=norm) 
        plt.xlabel('x', fontsize = 16, weight="bold")
        plt.ylabel('z', fontsize = 16, weight="bold")
        plt.xlim([xmin, xmax])
        plt.ylim([zmin, zmax])
        if with_dots:
            plt.plot([x_dots], [z_dots], marker='o', markersize=3, color="red")
            plt.plot(x_is, zmax - 20, marker='x', markersize=6, color='cyan')
            plt.plot(x_is, 0        , marker='x', markersize=6, color='cyan')
            plt.plot(x_is, zmin + 20, marker='x', markersize=6, color='cyan')
            plt.plot(x_le, zmax - 20, marker='+', markersize=6, color='cyan')
            plt.plot(x_le, 0        , marker='+', markersize=6, color='cyan')
            plt.plot(x_le, zmin + 20, marker='+', markersize=6, color='cyan')    

    plt.title(title, fontsize = 16, weight="bold")
#     plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal')
    ax = plt.gca()
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, cmap=cmap, norm=norm)
    cbar.ax.set_yticklabels(cbar_ticks) #, fontsize=16, weight='bold')
    cbar.set_label(label, rotation=270, fontsize = 16, weight="bold", labelpad=20)
    
    if (normales):
        for normale in normales:
            origin = normales[normale]['origin']
            vector = normales[normale]['vector']            
            if plane=='xy':                
                A = origin[0]
                B = origin[1]
                U = vector[0]
                V = vector[1]
            if plane=='xz':                
                A = origin[0]
                B = origin[2]
                U = vector[0]
                V = vector[2]
            ax.quiver(A, B, -U, V, scale=10)    # -U because of ax.invert_xaxis()
                        
        
    
    if (streamplot): 
        # Magnetic field lines
        
        if plane=='xy':
            ax.streamplot(X, Y, Bx.transpose(), Bj.transpose(), linewidth=linewidth, density=density)
        if plane=='xz':
            ax.streamplot(X, Z, Bx.transpose(), Bj.transpose(), linewidth=linewidth, density=density)

    ax.invert_xaxis()

    try:
        if save_dir and t_label:
            svg_name = t_label + title[0] + plane
            plt.savefig(save_dir+svg_name)
    except: 
        print("Please specify both save_dir and t_label")
        
    if not(loop):
        plt.show()
                  
    return ax


# ### Temporal_B

# In[20]:


def distance( pos1, pos2 ):
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return np.sqrt( (z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2 )


def find_closest_virtual_satellite(satellites, box_id, boxes=None):
    '''
    Given a dictionnary of virtual satellites of the format { 
                                                              sat_id_1 : { 'position': (x1, y1, z1),
                                                                           'other infos': xxx1      } ,
                                                              sat_id_2 : { 'position': (x2, y2, z2),
                                                                           'other infos': xxx2      } ,  
                                                              ...            
                                                                           
                                                             }
    and a box identifier (e.g. 'X_upstream'),
    this function returns the virtual satellite's ID which is the closest to the center of the box.                                                                     
    '''
    
    if not(boxes):
        boxes = create_boxes_dictionary()    
    
    distances = {}    
    for sat in satellites:
        
        pos_sat = ( satellites[sat]['position']['x'], 
                    satellites[sat]['position']['y'],
                    satellites[sat]['position']['z'] )            
        distances.update( {sat : distance(pos_sat, boxes[box_id]['center'])} )
        
    closest_sat = min(distances, key=distances.get)
    print(f"center of the '{box_id}' box: {boxes[box_id]['center']}")
    print(f"closest satellite's position: {satellites[closest_sat]['position']}")
    print('')       
        
    return closest_sat


# In[21]:


def update_satellites_with_satellite_info(satellites, file_satellite):
    
    with open(file_satellite , "r", encoding='utf-8') as f:

        content = f.read()

        infos = content.split()

        x_sat = int(float(infos[1]))
        y_sat = int(float(infos[2]))
        z_sat = int(float(infos[3]))

        sat_id = f"{x_sat}{y_sat}{z_sat}"

        satellites.update({sat_id : {"position" : {'x' : x_sat, 'y' : y_sat, 'z' : z_sat}}})

        infos = content.split('time')[1:]

        for info in infos:
            liste = info.split()        

            time = int(float(liste[0]))

            index_b = liste.index('B_field%xyz')

            B_field_x = liste[index_b + 1]
            B_field_y = liste[index_b + 2]
            B_field_z = liste[index_b + 3]

            satellites[sat_id].update({f't{time}': { 'Bx': B_field_x,
                                                     'By': B_field_y,
                                                     'Bz': B_field_z  }
                                       })
            
    return satellites         


# In[22]:


def plot_virtual_sats_positions(satellites, color='red', pre_figure=None):

    if not(pre_figure):
        fig = plt.figure(figsize=plt.figaspect(0.5)*1) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
        ax = fig.gca(projection='3d')
    else:
        fig = pre_figure[0]
        ax = pre_figure[1]

    for sat in satellites:
        x_sat = satellites[sat]['position']['x']
        y_sat = satellites[sat]['position']['y']
        z_sat = satellites[sat]['position']['z']
        ax.scatter(x_sat,y_sat,z_sat, marker='o', color=color)    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'{len(satellites)} virtual satellites')
    
    plt.show()
        
    return fig, ax


# In[23]:


def plot_temporal_B(satellite, title=None):
    
    '''
    This function needs a "satellite", of the form { t1: {'Bx': bx, 'By': by, 'Bz':bz},
                                                     t2: {'Bx': bx, 'By': by, 'Bz':bz},
                                                    ... }
    An easy way to get one is: satellites[sat_id], or satellites_boxes[box]
    '''
    
    time_vec = []
    Bx = []
    By = []
    Bz = []
       
    for t in satellite:
        if t=='position':
            continue
        time_vec = time_vec + [int(t[1:])]
        Bx = Bx + [float(satellite[t]['Bx'])]
        By = By + [float(satellite[t]['By'])]
        Bz = Bz + [float(satellite[t]['Bz'])]
        
    plt.figure()    
    if title:
        plt.title(title)
        
    plt.plot(time_vec, Bx, label='Bx')
    plt.plot(time_vec, By, label='By')
    plt.plot(time_vec, Bz, label='Bz')
    plt.ylabel('Magnetic field (nT)')
    plt.xlabel(r'time ($\Omega_{ci}^{-1}$)')
    
    plt.legend()
    plt.show()
            
    return time_vec, (Bx, By, Bz)


# In[ ]:




