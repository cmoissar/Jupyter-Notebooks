#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[1]:


from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

import import_ipynb
import Module_Diagnostics as MD


# # Plot settings

# In[2]:


rcParams["figure.figsize"] = [10.4, 4.8]


# # Choose the story

# ## Choose the run

# In[3]:


Cluster = 'Zoidberg'
Cluster = 'Curie'
Cluster = 'Occ'

''''''''''''''''''''''''''''''''''''''''''''''''
run_name = '20_04_23_very_rough_big_one_Bz-By-_re'
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
run_name = '20_05_17_big_one_0'
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
run_name = '20_08_18_new_big_one_0'
''''''''''''''''''''''''''''''''''''''''''''''''

#Infos about this run
t_start_SW = 180
t_start_Sh = 217
t_start_MC = 245

t_Sh_at_x0 = 223

storing_directory = f"../{Cluster}/{run_name}/json_files/"
name = "sav_story_" + run_name + ".json"
path = Path(storing_directory + name)


# ## Choose the stories

# In[4]:


#Stories dictionnary:
STORIES = {}

#Stories size
type_of_story_size = 'Magnetosheath size'
stories_to_tell_size = ('size_nose', 'size_yup', 'size_ydown', 'size_zup', 'size_zdown')
unit_size = r'$d_i$'
STORIES.update({type_of_story_size: {'stories': stories_to_tell_size, 'unit': unit_size}})

type_of_story_bow_shock = 'Position of the bow shock'
stories_to_tell_bow_shock = ('x_bow_shock',
                             'y_bow_shock_up', 'y_bow_shock_down',
                             'z_bow_shock_up', 'z_bow_shock_down')
STORIES.update({type_of_story_bow_shock: {'stories': stories_to_tell_bow_shock, 'unit': unit_size}})

type_of_story_magnetopause = 'Position of the Magnetopause'
stories_to_tell_magnetopause = ('x_magnetopause',
                                'y_magnetopause_up', 'y_magnetopause_down',
                                'z_magnetopause_up', 'z_magnetopause_down')
STORIES.update({type_of_story_magnetopause: {'stories': stories_to_tell_magnetopause, 'unit': unit_size}})

#Stories others
boxes = ('upstream', 'nose', 'yup', 'ydown', 'zup', 'zdown')

#Plasma parameters
#Stories N
type_of_story_N = 'Density'
stories_to_tell_N = [f'N_{box}' for box in boxes]
unit_N = r'cm$^{-3}$'
STORIES.update({type_of_story_N: {'stories': stories_to_tell_N, 'unit': unit_N}})

#Stories V
type_of_story_V = 'Plasma velocity'
stories_to_tell_V = [f'V_{box}' for box in boxes]
unit_V = r'km/s'
STORIES.update({type_of_story_V: {'stories': stories_to_tell_V, 'unit': unit_V}})

#Stories T
type_of_story_T = 'Proton temperature'
stories_to_tell_T = [f'T_{box}' for box in boxes]
unit_T = r'K'
STORIES.update({type_of_story_T: {'stories': stories_to_tell_T, 'unit': unit_T}})

#Fields
#Stories B
type_of_story_B = 'Magnetic Field'
stories_to_tell_B = [f'B_{box}' for box in boxes]
unit_B = r'nT'
STORIES.update({type_of_story_B: {'stories': stories_to_tell_B, 'unit': unit_B}})

type_of_story_Bz = 'Magnetic Field along z'
stories_to_tell_Bz = [f'Bz_{box}' for box in boxes]
STORIES.update({type_of_story_Bz: {'stories': stories_to_tell_Bz, 'unit': unit_B}})

#Stories E
type_of_story_E = 'Electric Field'
stories_to_tell_E = [f'E_{box}' for box in boxes]
unit_E = r'mV/m'
STORIES.update({type_of_story_E: {'stories': stories_to_tell_E, 'unit': unit_E}})

type_of_story_Ey = 'Driving electric field (Ey)'
stories_to_tell_Ey = [f'Ey_{box}' for box in boxes]
STORIES.update({type_of_story_Ey: {'stories': stories_to_tell_Ey, 'unit': unit_E}})

#Calculated
#Stories J
type_of_story_J = 'Current'
stories_to_tell_J = [f'J_{box}' for box in boxes]
unit_J = r'nA/m²'
STORIES.update({type_of_story_J: {'stories': stories_to_tell_J, 'unit': unit_J}})

#Stories rmsB
type_of_story_rmsB = 'Turbulence #1 (rmsB)'
stories_to_tell_rmsB = [f'rmsB_{box}' for box in boxes]
unit_rmsB = r'nT'
STORIES.update({type_of_story_rmsB: {'stories': stories_to_tell_rmsB, 'unit': unit_rmsB}})

type_of_story_rmsBoB = 'Turbulence #2 (rmsBoB)'
stories_to_tell_rmsBoB = [f'rmsBoB_{box}' for box in boxes]
unit_rmsBoB = r''
STORIES.update({type_of_story_rmsBoB: {'stories': stories_to_tell_rmsBoB, 'unit': unit_rmsBoB}})

#stories J.E
type_of_story_JE = 'Joule heating (J.E)'
stories_to_tell_JE = [f'JE_{box}' for box in boxes]
unit_JE = r'pW/m$^3$'     #'nA/m² . mV/m'
STORIES.update({type_of_story_JE: {'stories': stories_to_tell_JE, 'unit': unit_JE}})

#stories Pressure
type_of_story_Pmag = 'Magnetic pressure'
stories_to_tell_Pmag = [f'Pmag_{box}' for box in boxes]
unit_P = r'nPa'
STORIES.update({type_of_story_Pmag: {'stories': stories_to_tell_Pmag, 'unit': unit_P}})

type_of_story_Pdyn = 'Dynamic pressure'
stories_to_tell_Pdyn = [f'Pdyn_{box}' for box in boxes]
STORIES.update({type_of_story_Pdyn: {'stories': stories_to_tell_Pdyn, 'unit': unit_P}})

type_of_story_Pth = 'Thermal pressure'
stories_to_tell_Pth = [f'Pth_{box}' for box in boxes]
STORIES.update({type_of_story_Pth: {'stories': stories_to_tell_Pth, 'unit': unit_P}})

#stories Beta
type_of_story_Beta = 'Plasma Beta'
stories_to_tell_Beta = [f'Beta_{box}' for box in boxes]
unit_Beta = ''
STORIES.update({type_of_story_Beta: {'stories': stories_to_tell_Beta, 'unit': unit_Beta}})


# # Read the story

# ### Load data

# In[5]:


with open(path, "r", encoding='utf-8') as story:
    data = json.load(story)


# In[6]:


data['t00150']['JxB_nose']


# In[7]:


data['t00215']


# ### Def plot functions

# In[8]:


def select_color(chosen_story):
    color = '0.7'
    
    if ('nose' in chosen_story or 'x_' in chosen_story):
        color = 'RoyalBlue'
    if ('upstream') in chosen_story:
        color = 'Orchid'
    if (chosen_story.startswith('y') and ('_up' in chosen_story)) or ('yup' in chosen_story):    
        color = 'Red'
    if (chosen_story.startswith('y') and ('_down' in chosen_story)) or ('ydown' in chosen_story):    
        color = 'Orange'
    if (chosen_story.startswith('z') and ('_up' in chosen_story)) or ('zup' in chosen_story):    
        color = 'Green'
    if (chosen_story.startswith('z') and ('_down' in chosen_story)) or ('zdown' in chosen_story):    
        color = 'MediumSeaGreen'
    return color

def tell_story(chosen_story, include_cloud=False, only_positive=False, with_lines=False):

    T = []
    Y = []
    
    for time_dump in sorted(data):
        
        if int(time_dump[1:]) > t_start_SW:
            try:
                if only_positive:
                    if isinstance(data[time_dump][chosen_story], list):
                        Y.append(MD.norm(data[time_dump][chosen_story]))
                    else:
                        Y.append(abs(data[time_dump][chosen_story]))
                else:
                    if isinstance(data[time_dump][chosen_story], list):
                        Y.append(MD.norm(data[time_dump][chosen_story]))
                    else:
                        Y.append(data[time_dump][chosen_story])
                T.append(int(time_dump.strip('t'))) 
            except KeyError:
                print(f"{chosen_story} for time {time_dump} has not been written yet")    
            if ( (include_cloud == False) and (int(time_dump[1:]) >= t_start_MC+10) ):
                break

    color = select_color(chosen_story)
    plt.scatter(T,Y, color=color, label=chosen_story)
    if with_lines:
        plt.plot(T,Y, linestyle='--', color=color)
    return

def tell_group_of_stories(stories_to_tell, type_of_story, unit, include_cloud = False, only_positive=False):
    
    plt.figure()
    for story in stories_to_tell:
        tell_story(story, include_cloud, only_positive)  
    plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
    plt.ylabel(type_of_story + f' ({unit})')

    # get current xmin, xmax, ymin, ymax
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()    
    plt.ylim(ymin, ymax*1.1)
    ymin, ymax = plt.ylim() 
    
    plt.axvspan(xmin/2    , t_start_Sh  , facecolor='blue', alpha=0.1)
    plt.axvspan(t_start_Sh, t_start_MC  , facecolor='red', alpha=0.1)
    plt.axvspan(t_Sh_at_x0, t_start_MC  , facecolor='red', alpha=0.1)
    plt.axvspan(t_start_MC, xmax*2      , facecolor='yellow', alpha=0.2)
    
    # reset xmin, xmax to previous values. axvspan changed them and it is ugly
    plt.xlim(xmin, xmax)
     
    name_region = 'Solar Wind'
    plt.text(xmin+(t_start_Sh-xmin)/2-(len(name_region)+2)/2,
             ymin + 0.935*(ymax-ymin), name_region,
             bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 4})
    
    name_region = 'Sheath'
    plt.text(t_start_Sh+(t_start_MC-t_start_Sh)/2-(len(name_region)+2)/2,
             ymin + 0.935*(ymax-ymin), name_region,
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 4})
    
    name_region = 'Magnetic Cloud'
    plt.text(t_start_MC+(xmax-t_start_MC)/2-(len(name_region)+2)/2,
             ymin + 0.935*(ymax-ymin), name_region,
             bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 4})
         
    plt.title(type_of_story, size = 'large', weight = 'bold')
#     plt.legend(loc='center left') #, bbox_to_anchor=(0.5, 0.5))    
    
    axe = plt.gca()    
    # Shrink current axis by 20%
    box = axe.get_position()
    axe.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    axe.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    
    plt.show()
    
    return


# ### Plots

# In[9]:


plt.close('all')
for group_of_stories in STORIES:
    tell_group_of_stories(stories_to_tell = STORIES[group_of_stories]['stories'],
                          type_of_story = group_of_stories,
                          unit = STORIES[group_of_stories]['unit'])


# ### Magnetosheath size evolution

# #### #1 Multivariate plot (x-direction)

# In[10]:


x_bs_list = []
x_mp_list = []
x_ip_shock_list = []
x_mc_leading_edge_list = []
time_list = []
deltaP_list = []

for time in sorted(data):
    time_list.extend([int(time[1:])])
    x_bs_list.extend([data[time]['x_bow_shock']])
    x_mp_list.extend([data[time]['x_magnetopause']])    
    x_ip_shock_list.extend([data[time]['x_ip_shock']])
    x_mc_leading_edge_list.extend([data[time]['x_mc_leading_edge']])
    deltaP_list.extend([ data[time]['Pmag_nose']    +data[time]['Pth_nose']    +data[time]['Pdyn_nose']
                        -data[time]['Pmag_upstream']+data[time]['Pth_upstream']+data[time]['Pdyn_upstream']])


# In[11]:


print(x_ip_shock_list)
print(x_mc_leading_edge_list)


# In[12]:


# This is a safeguard. However if metadata in Story_Writer.ipynb is well defined, this shouldn't be necessary

x_is = [(time_list[0],x_ip_shock_list[0])]
x_is = [ (time_list[i], x_ip_shock_list[i]) 
         for i in range(0, len(x_ip_shock_list))
         if (i==0 or (x_ip_shock_list[i] < min([x[1] for x in x_is]))) ]

x_le = [(time_list[0],x_mc_leading_edge_list[0])]
x_le = [ (time_list[i], x_mc_leading_edge_list[i])
         for i in range(0, len(x_mc_leading_edge_list))
         if (i==0 or x_mc_leading_edge_list[i] < min([x[1] for x in x_le])) ]

print(x_is)
print(x_le)


# In[13]:


from scipy.interpolate import interp1d
x_bs_smooth = interp1d(time_list, x_bs_list, kind='cubic')
x_mp_smooth = interp1d(time_list, x_mp_list, kind='cubic')
x_is_smooth = interp1d([t[0] for t in x_is],  [x[1] for x in x_is], kind='cubic')
x_le_smooth = interp1d([t[0] for t in x_le],  [x[1] for x in x_le], kind='cubic')
time_smooth = np.linspace(time_list[0], time_list[-1], num=1001, endpoint=True)
x_bs_smooth = x_bs_smooth(time_smooth)
x_mp_smooth = x_mp_smooth(time_smooth)
t_is_smooth = np.linspace([t[0] for t in x_is][0], [t[0] for t in x_is][-1], num=1001, endpoint=True)
x_is_smooth = x_is_smooth(t_is_smooth)
t_le_smooth = np.linspace([t[0] for t in x_le][0], [t[0] for t in x_le][-1], num=1001, endpoint=True)
x_le_smooth = x_le_smooth(t_le_smooth)


# #### Old

# In[14]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x_bs_smooth, time_smooth, linewidth=2, label='bow shock')
plt.plot(x_mp_smooth, time_smooth, linewidth=2, label='magnetopause')
# plt.scatter(x_ip_shock_list, time_list, label='ip_shock')
plt.plot(x_is_smooth, t_is_smooth, linewidth=2, label='ip_shock')
# plt.scatter(x_mc_leading_edge_list, time_list, label='mc_leading_edge')
plt.plot(x_le_smooth, t_le_smooth, linewidth=2, label='mc_leading_edge')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('distance from Earth (di)', weight = 'bold')
plt.ylabel('time $(\Omega_{ci}^{-1})$', weight = 'bold', rotation=270)
ax.yaxis.set_label_coords(-0.1,0.5)
plt.legend()

import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

rectangles = []
colors = []

for i in range(150,len(time_smooth)):
    
    try:
        j = np.where(time_list < time_smooth[i])[0][-1]
    except IndexError: 
        j = j
    
    try:
        hauteur = time_smooth[i+1] - time_smooth[i]
    except IndexError:
        hauteur = np.mean(np.array(time_smooth[1:]) - np.array(time_smooth[:-1]))
            
    rectangle = patches.Rectangle( (x_mp_smooth[i],time_smooth[i])  , #position of the bottom-left corner (or top-right if axes are inverted)
                                    x_bs_smooth[i] - x_mp_smooth[i] , #largeur
                                    hauteur                         )
                                  
    rectangles.append(rectangle)
    colors.append(deltaP_list[j])
       
colors = np.array(colors)
p = PatchCollection(rectangles, cmap=matplotlib.cm.Reds)
p.set_array(colors)
ax.add_collection(p)
plt.colorbar(p, label = "$\Delta$P " + f"({unit_P})")
plt.xlim([1.2*max(x_bs_smooth), 0])

plt.scatter(x_bs_list, time_list, label='bow shock') 
plt.scatter(x_mp_list, time_list, label='magnetopause')

plt.show()


# #### New

# In[15]:


from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x_bs_list, time_list, label='bow shock', color=cm.plasma(np.array(deltaP_list)/max(deltaP_list))) 
plt.scatter(x_mp_list, time_list, label='magnetopause')
plt.plot(x_is_smooth, t_is_smooth, linewidth=2, label='ip_shock', color='green')
plt.plot(x_le_smooth, t_le_smooth, linewidth=2, label='mc_leading_edge', color='red')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('distance from Earth (di)', weight = 'bold')
plt.ylabel('time $(\Omega_{ci}^{-1})$', weight = 'bold', rotation=270)
ax.yaxis.set_label_coords(-0.1,0.5)
plt.legend()

import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

rectangles = []
colors = []

for i in range(150,len(time_smooth)):
    
    try:
        j = np.where(time_list < time_smooth[i])[0][-1]
    except IndexError: 
        j = j
    
    try:
        hauteur = time_smooth[i+1] - time_smooth[i]
    except IndexError:
        hauteur = np.mean(np.array(time_smooth[1:]) - np.array(time_smooth[:-1]))
            
    rectangle = patches.Rectangle( (x_mp_smooth[i],time_smooth[i])  , #position of the bottom-left corner (or top-right if axes are inverted)
                                    x_bs_smooth[i] - x_mp_smooth[i] , #largeur
                                    hauteur                         )
                                  
    rectangles.append(rectangle)
    colors.append(deltaP_list[j])
        
colors = []    
for j in range(150, len(time_list)):
    colors.append(deltaP_list[j])
       
colors = np.array(colors)
p = PatchCollection(rectangles, cmap=matplotlib.cm.plasma)
p.set_array(colors)
ax.add_collection(p)
plt.colorbar(p, label = "$\Delta$P " + f"({unit_P})")
plt.xlim([1.2*max(x_bs_smooth), 0])
plt.ylim([280,190])

#add background color
ax.set_facecolor('lemonchiffon')

plt.show()


# In[16]:


get_ipython().run_line_magic('matplotlib', '')
plt.scatter(time_list, x_bs_list)


# In[17]:


np.shape(time_list)
np.shape(x_bs_list)


# #### #2 First story y & z directions

# In[18]:


STORIES_2 = {}

type_of_story_bs_mp_flanks = 'Position of the bow shock & magnetopause on the flanks'
stories_to_tell_bs_mp_flanks = ('y_bow_shock_up', 'y_bow_shock_down',
                                'z_bow_shock_up', 'z_bow_shock_down',
                                'y_magnetopause_up', 'y_magnetopause_down',
                                'z_magnetopause_up', 'z_magnetopause_down')
STORIES_2.update({type_of_story_bs_mp_flanks: {'stories': stories_to_tell_bs_mp_flanks, 'unit': unit_size}})


# In[19]:


plt.close('all')
for group_of_stories in STORIES_2:
    tell_group_of_stories(stories_to_tell = STORIES_2[group_of_stories]['stories'],
                          type_of_story = group_of_stories,
                          unit = STORIES_2[group_of_stories]['unit'],
                          only_positive = True,
                          include_cloud = False)


# ### Comparison to real data

# #### Turbulence

# In[20]:


for rmsB in STORIES['Turbulence #1 (rmsB)']['stories']:
    print(rmsB)
    for t in data:
        print(f'{t} : {data[t][rmsB]:.2f}')


# In[21]:


STORIES['Turbulence #1 (rmsB)']['stories']


# In[22]:


for rmsBoB in STORIES['Turbulence #2 (rmsBoB)']['stories']:
    print(rmsBoB)
    for t in data:
        print(f'{t} : {data[t][rmsBoB]:.2f}')


# #### Velocity

# In[23]:


# Estimation of V_A and V_MS
b = 1e-9
n = 1e6
v = 1e3
t = 11605

def V_Alfven(box, time):
    '''
    box in ['upstream', 'nose', 'yup', 'ydown', 'zup', 'zdown']
    '''
    B = data[time][f'B_{box}']
    N = data[time][f'N_{box}']
    v_A = np.sqrt((b*B)**2 / (MD.mp * (n*N) * MD.µ0))
    return v_A/v #(km/s)

def C_sound(box, time):
    '''
    box in ['upstream', 'nose', 'yup', 'ydown', 'zup', 'zdown']
    '''
    T = data[time][f'T_{box}']
    c_s = np.sqrt(MD.kB * t*T / MD.mp)
    return c_s/v #(km/s)

def V_fast(box, time):
    '''
    box in ['upstream', 'nose', 'yup', 'ydown', 'zup', 'zdown']
    '''
    v_ms = np.sqrt(V_Alfven(box, time)**2+C_sound(box, time)**2)
    return v_ms #(km/s)


# In[24]:


# t_quiet = 't00180'
# box = 'zdown'

# V_fast(box, t_quiet) < data[t_quiet][f'V_{box}']


# In[25]:


# print(f"V_{box} = {data[t_quiet][f'V_{box}']:.0f} km/s")


# In[26]:


# print(f"C_sound = {C_sound(box, t_quiet):.0f} km/s")
# print(f"V_Alfven = {V_Alfven(box, t_quiet):.0f} km/s")
# print(f"V_fast = {V_fast(box, t_quiet):.0f} km/s")


# In[27]:


v_A = np.sqrt((b*47)**2 / (MD.mp * (n*73) * MD.µ0)) / v
print(430/v_A)


# #### Pressure

# In[28]:


# box = 'upstream'

# print(f"P_total = {data[t_quiet][f'Pth_{box}']+data[t_quiet][f'Pdyn_{box}']+data[t_quiet][f'Pmag_{box}'] :.2f} nPa")


# #### Beta

# In[29]:


# t_quiet = 't00180'
# box = 'nose'

# data[t_quiet][f'Beta_{box}']


# #### Rankine-Hugoniot

# ##### Gather

# In[30]:


time_vec = []

N_upstream, N_nose = ([], [])

Vx_upstream = []
Vx_nose = []
Vy_upstream = []
Vy_nose = []
Vz_upstream = []
Vz_nose = []

Bx_upstream = []
Bx_nose = []
By_upstream = []
By_nose = []
Bz_upstream = []
Bz_nose = []

Pth_upstream = []
Pth_nose = []
Pdyn_upstream = []
Pdyn_nose = []
Pmag_upstream = []
Pmag_nose = []


for time_label in data:
    
    time_vec = time_vec + [ int(time_label[1:]) ]
    
    N_upstream = N_upstream + [ data[time_label]['N_upstream'] ]
    N_nose = N_nose + [ data[time_label]['N_nose'] ]
    
    Vx_upstream = Vx_upstream + [ data[time_label]['Vx_upstream'] ]
    Vx_nose = Vx_nose + [ data[time_label]['Vx_nose'] ]
    Vy_upstream = Vy_upstream + [ data[time_label]['Vy_upstream'] ]
    Vy_nose = Vy_nose + [ data[time_label]['Vy_nose'] ]
    Vz_upstream = Vz_upstream + [ data[time_label]['Vz_upstream'] ]
    Vz_nose = Vz_nose + [ data[time_label]['Vz_nose'] ]
        
    Bx_upstream = Bx_upstream + [ data[time_label]['Bx_upstream'] ]
    Bx_nose = Bx_nose + [ data[time_label]['Bx_nose'] ]
    By_upstream = By_upstream + [ data[time_label]['By_upstream'] ]
    By_nose = By_nose + [ data[time_label]['By_nose'] ]
    Bz_upstream = Bz_upstream + [ data[time_label]['Bz_upstream'] ]
    Bz_nose = Bz_nose + [ data[time_label]['Bz_nose'] ]
    
    Pth_upstream = Pth_upstream + [ data[time_label]['Pth_upstream'] ]
    Pth_nose = Pth_nose + [ data[time_label]['Pth_nose'] ]
    Pdyn_upstream = Pdyn_upstream + [ data[time_label]['Pdyn_upstream'] ]
    Pdyn_nose = Pdyn_nose + [ data[time_label]['Pdyn_nose'] ]
    Pmag_upstream = Pmag_upstream + [ data[time_label]['Pmag_upstream'] ]
    Pmag_nose = Pmag_nose + [ data[time_label]['Pmag_nose'] ]


# ##### $$ [B_x] = 0 $$

# In[31]:


plt.close()
plt.plot(time_vec, Bx_upstream, label = 'Bx_upstream')
plt.plot(time_vec, Bx_nose, label = 'Bx_nose')
plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nT)')
plt.legend()
plt.show()


# #####  $$ [B_y V_x - B_x V_y] = 0 $$

# In[32]:


plt.close()

plt.plot(time_vec,
         [By_upstream[i]*Vx_upstream[i] - Bx_upstream[i]*Vy_upstream[i] for i in range(0, len(time_vec))],
         label = '(By Vx - Bx Vy)_upstream')

plt.plot(time_vec,
         [By_nose[i]*Vx_nose[i] - Bx_nose[i]*Vy_nose[i] for i in range(0, len(time_vec))],
         label = '(By Vx - Bx Vy)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nT . km/s)')
plt.legend()
plt.show()


# ##### $$ [B_z V_x - B_x V_z] = 0 $$

# In[33]:


plt.close()

plt.plot(time_vec,
         [Bz_upstream[i]*Vx_upstream[i] - Bx_upstream[i]*Vz_upstream[i] for i in range(0, len(time_vec))],
         label = '(Bz Vx - Bx Vz)_upstream')

plt.plot(time_vec,
         [Bz_nose[i]*Vx_nose[i] - Bx_nose[i]*Vz_nose[i] for i in range(0, len(time_vec))],
         label = '(Bz Vx - Bx Vz)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nT . km/s)')
plt.legend()
plt.show()


# ##### $$ [N V_x] = 0 $$

# In[34]:


plt.close()

plt.plot(time_vec,
         [N_upstream[i]*Vx_upstream[i] for i in range(0, len(time_vec))],
         label = '(N Vx)_upstream')

plt.plot(time_vec,
         [N_nose[i]*Vx_nose[i] for i in range(0, len(time_vec))],
         label = '(N Vx)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (m^-3 . km/s)')
plt.legend()
plt.show()


# ##### $$ [P_{th} + P_{dyn} + P_{mag}] = 0 $$

# In[35]:


plt.close()

plt.plot(time_vec,
         [ Pth_upstream[i] + Pdyn_upstream[i] + Pmag_upstream[i] for i in range(0, len(time_vec)) ],
         label = '(Pth + Pmag + Pdyn)_upstream')

plt.plot(time_vec,
         [ Pth_nose[i] + Pdyn_nose[i] + Pmag_nose[i] for i in range(0, len(time_vec)) ],
         label = '(Pth + Pmag + Pdyn)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nPa)')
plt.legend()
plt.show()


# ##### $$ [ \rho V_x V_y - B_x B_y / \mu_0] = 0 $$ 

# In[36]:


plt.close()

v = 1e3
b = 1e-9
p = 1e9

plt.plot(time_vec,
         [  MD.mp * Vx_upstream[i] * Vy_upstream[i] * v**2 * p
          - Bx_upstream[i] * By_upstream[i] / MD.µ0 * b**2 * p for i in range(0, len(time_vec)) ],
         label = '(rho Vx Vy - Bx By / µ0)_upstream')

plt.plot(time_vec,
         [  MD.mp * Vx_nose[i] * Vy_nose[i] * v**2 * p
          - Bx_nose[i] * By_nose[i] / MD.µ0 * b**2 * p for i in range(0, len(time_vec)) ],
         label = '(rho Vx Vy - Bx By / µ0)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nPa)')
plt.legend()
plt.show()


# ##### $$ [ \rho V_x V_z - B_x B_z / \mu_0] = 0 $$ 

# In[37]:


plt.close()

v = 1e3
b = 1e-9
p = 1e9

plt.plot(time_vec,
         [  MD.mp * Vx_upstream[i] * Vz_upstream[i] * v**2 * p
          - Bx_upstream[i] * Bz_upstream[i] / MD.µ0 * b**2 * p for i in range(0, len(time_vec)) ],
         label = '(rho Vx Vz - Bx Bz / µ0)_upstream')

plt.plot(time_vec,
         [  MD.mp * Vx_nose[i] * Vz_nose[i] * v**2 * p
          - Bx_nose[i] * Bz_nose[i] / MD.µ0 * b**2 * p for i in range(0, len(time_vec)) ],
         label = '(rho Vx Vz - Bx Bz / µ0)_nose')

plt.xlabel(r"time ($\Omega_{ci}^{-1}$)")
plt.ylabel(f' (nPa)')
plt.legend()
plt.show()


# # Temporal_B

# In[38]:


data_quiet = {}
for t in data:
    if int(t[1:]) <= 190: #warning: ugly manual value
        data_quiet.update({t: {'B_upstream': data[t]['B_upstream'],
                               'V_upstream': data[t]['V_upstream'],
                               'N_upstream': data[t]['N_upstream']}})
        
B0 = np.mean([data_quiet[t]['B_upstream'] for t in data_quiet])
N0 = np.mean([data_quiet[t]['N_upstream'] for t in data_quiet])
#Alfven speed is used to normalise velocities in Lathys
V0 = B0*MD.b / np.sqrt(N0*MD.n*MD.mp*MD.µ0) / MD.v 

print(B0, V0, N0)


# In[39]:


B_time_directory = '../' + Cluster + '/' + run_name + '/temporal_B/'
satellites = {}


# In[40]:


for filename in os.listdir(B_time_directory):
    
    file = B_time_directory + filename 
    satellites = MD.update_satellites_with_satellite_info(satellites, file, B0, V0, N0)


# In[41]:


# with open(file , "r", encoding='utf-8') as f:

#     content = f.read()
#     infos = content.split()
#     infos = content.split('time')[1:]    
    
#     for info in infos:
#         liste = info.split()

#         index_b = liste.index('B_field%xyz')
#         index_v = liste.index('velocity%xyz')
#         index_n = liste.index('density')

#         B_field_x = liste[index_b + 1] * B0


# In[42]:


time_of_interest = 't00195'

boxes = data[time_of_interest]['boxes']

satellites_boxes = {}

for box in boxes:
    closest_sat = MD.find_closest_virtual_satellite(satellites, box, boxes)
    satellites_boxes.update({box : satellites[closest_sat]})


# In[43]:


#choose a satellite far away from the magnetosheath
sat_id = '03000'
satellites_boxes.update({'unperturbed event' : satellites[sat_id]})
print("unperturbed event's satellite position:", satellites[sat_id]['position'] )


# In[44]:


#choose a satellite that will cross the bow shock twice
sat_id = '5000'
satellites_boxes.update({'two crossings' : satellites[sat_id]})
print("two bow shock crossings' satellite position:", satellites[sat_id]['position'] )


# In[45]:


# %matplotlib notebook
fig, ax = MD.plot_virtual_sats_positions(satellites)

MD.plot_virtual_sats_positions(satellites_boxes, color='blue', pre_figure=(fig, ax))

yy, zz = np.meshgrid(range(-50, 50), range(-50, 50))
# ax.plot_surface(boxes['nose']['coord_bow_shock'], yy, zz,  alpha=0.8)

plt.show()


# In[46]:


# plt.close('all')
# #Note: not the best way to think about it. The boxes change position during the simulation...
# for box in boxes:    
#     MD.plot_temporal_B(satellites_boxes[box], title = box + f" at {satellites_boxes[box]['position']}")


# In[47]:


box = 'unperturbed event'

time_vec, (Bx, By, Bz), (Vx, Vy, Vz), N = MD.plot_temporal_B(satellites_boxes[box],
                                                             title = box + f" at {satellites_boxes[box]['position']}")

B = np.array([np.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2) for i in range(0, len(Bx))])
V = np.array([np.sqrt(Vx[i]**2 + Vy[i]**2 + Vz[i]**2) for i in range(0, len(Vx))])


# In[48]:


Bx_SM = np.mean(Bx[177:197])
Bx_SH = np.mean(Bx[227:232])
By_SM = np.mean(By[177:197])
By_SH = np.mean(By[227:232])
Bz_SM = np.mean(Bz[177:197])
Bz_SH = np.mean(Bz[227:232])

Vx_SM = np.mean(Vx[177:197])
Vx_SH = np.mean(Vx[227:232])
Vy_SM = np.mean(Vy[177:197])
Vy_SH = np.mean(Vy[227:232])
Vz_SM = np.mean(Vz[177:197])
Vz_SH = np.mean(Vz[227:232])

N_SM = np.mean(N[177:197])
N_SH = np.mean(N[227:232])


# In[49]:


Vz_SH


# In[50]:


Vx_SH * Bz_SH / Bx_SH


# In[51]:


np.mean((Vx * Bz / Bx)[227:232])


# In[52]:


box = 'upstream'

time_vec, (Bx, By, Bz), (Vx, Vy, Vz), N = MD.plot_temporal_B(satellites_boxes[box], 
                                                             title = box + f" at {satellites_boxes[box]['position']}")

B = [np.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2) for i in range(0, len(Bx))]


# In[53]:


box = 'nose'

time_vec, (Bx, By, Bz), (Vx, Vy, Vz), N = MD.plot_temporal_B(satellites_boxes[box], 
                                                             title = box + f" at {satellites_boxes[box]['position']}")

B = [np.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2) for i in range(0, len(Bx))]


# In[54]:


box = 'two crossings'

time_vec, (Bx, By, Bz), (Vx, Vy, Vz), N = MD.plot_temporal_B(satellites_boxes[box], 
                                                             title = box + f" at {satellites_boxes[box]['position']}")

B = [np.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2) for i in range(0, len(Bx))]


# ### Spectra & Cie

# In[55]:


boxes[box]['box_indexes']['xmax'] - boxes[box]['box_indexes']['xmin']

#next: translate that to time. Then define rmsBoB based on this.


# In[ ]:


import pylab as plt
import numpy as np

import os
import glob
import re

from calc.vector_algebra import norm, dot_product, cross_product
from calc.Wavelets import calc_Morlet
from matplotlib import rc


# In[ ]:


n_avg = 5

def moving_average(a, n = n_avg) :
    
    avg = np.zeros(len(a))
    
    avg[:n] = np.nan
    avg[-n:] = np.nan
    
    for i in range(n, len(a)-n):
        avg[i] = np.mean(a[i-n:i+n])
    
    return avg

def fluct_maker(pos, Ax, Ay, Az, avg_type):

    #TODO : add option for linear fit or sliding window
    
    if avg_type == 'linear' :
        # Linear Fit'
        m, b = np.polyfit(pos, Ax, 1)
        Ax0 = m * pos + b
        m, b = np.polyfit(pos, Ay, 1)
        Ay0 = m * pos + b
        m, b = np.polyfit(pos, Az, 1)
        Az0 = m * pos + b

    if avg_type == 'slide':

        Ax0 = moving_average(Ax)[n_avg:-n_avg]
        Ay0 = moving_average(Ay)[n_avg:-n_avg]
        Az0 = moving_average(Az)[n_avg:-n_avg]

        Ax = Ax[n_avg:-n_avg]
        Ay = Ay[n_avg:-n_avg]
        Az = Az[n_avg:-n_avg]

    A0 = np.sqrt(Ax0 ** 2 + Ay0 ** 2 + Az0 ** 2)

    dAx = Ax - Ax0
    dAy = Ay - Ay0
    dAz = Az - Az0

    e1 = np.ones(len(Ax0))
    e0 = np.zeros(len(Ax0))

    # e_para = dot_product()
    e_para_x = Ax0 / A0
    e_para_y = Ay0 / A0
    e_para_z = Az0 / A0

    e_perp1 = cross_product(e1, e0, e0, Ax0, Ay0, Az0)
    e_perp1 = e_perp1 / norm(e_perp1)
    e_perp1_x = e_perp1[0]
    e_perp1_y = e_perp1[1]
    e_perp1_z = e_perp1[2]

    e_perp2 = cross_product(e_perp1_x, e_perp1_y, e_perp1_z, Ax0, Ay0, Az0)
    e_perp2 = e_perp2 / norm(e_perp2)
    e_perp2_x = e_perp2[0]
    e_perp2_y = e_perp2[1]
    e_perp2_z = e_perp2[2]

    dA_para = dot_product(dAx, dAy, dAz, e_para_x, e_para_y, e_para_z)
    dA_perp1 = dot_product(dAx, dAy, dAz, e_perp1_x, e_perp1_y, e_perp1_z)
    dA_perp2 = dot_product(dAx, dAy, dAz, e_perp2_x, e_perp2_y, e_perp2_z)

    return dAx, dAy, dAz, dA_para, dA_perp1, dA_perp2, A0, Ax0, Ay0, Az0

def spectra_maker(pos, Ax, Ay, Az, avg_type):

    dAx, dAy, dAz, dA_para, dA_perp1, dA_perp2, A0, Ax0, Ay0, Az0 = fluct_maker(pos, Ax, Ay, Az, avg_type)
    
    dt = np.mean(pos[1:] - pos[:-1])

    Morlet_Ax = calc_Morlet(dt, dAx)
    Morlet_Ay = calc_Morlet(dt, dAy)
    Morlet_Az = calc_Morlet(dt, dAz)

    PSD_tot = Morlet_Ax.PSD + Morlet_Ay.PSD + Morlet_Az.PSD

    Morlet_para = calc_Morlet(dt, dA_para)
    Morlet_perp1 = calc_Morlet(dt, dA_perp1)
    Morlet_perp2 = calc_Morlet(dt, dA_perp2)

    power = Morlet_Ax.power + Morlet_Ay.power + Morlet_Az.power
    powerHz = Morlet_Ax.powerHz + Morlet_Ay.powerHz + Morlet_Az.powerHz

    power_para = Morlet_para.power
    powerHz_para = Morlet_para.powerHz
    power_perp = Morlet_perp1.power + Morlet_perp2.power
    powerHz_perp = Morlet_perp1.powerHz + Morlet_perp2.powerHz

    Anisotropy = powerHz_perp / (2 * powerHz_para)

    PSD_para = Morlet_para.PSD
    PSD_perp = Morlet_perp1.PSD + Morlet_perp2.PSD

    T, S = np.meshgrid(Morlet_Ax.time, Morlet_Ax.scales)

    return T, S, PSD_tot, PSD_para, PSD_perp


# In[ ]:


'''Solar wind'''

t_SW_start = 130
t_SW_end   = 150

t_SW = np.array([t for t in range(t_SW_start, t_SW_end, 1)])

PSD_tot_SW_list  = []
PSD_para_SW_list = []
PSD_perp_SW_list = []

'''Sheath'''

t_Sh_start = 215
t_Sh_end   = 235
        
t_Sh = np.array([t for t in range(t_Sh_start, t_Sh_end, 1)])

PSD_tot_Sh_list = []
PSD_para_Sh_list = []
PSD_perp_Sh_list = []

'''Magnetic Cloud'''

t_MC_start = 255-n_avg
t_MC_end   = 275+n_avg
        
t_MC = np.array([t for t in range(t_MC_start+n_avg, t_MC_end-n_avg,1 )])

PSD_tot_MC_list = []
PSD_para_MC_list = []
PSD_perp_MC_list = []


# In[ ]:


B_SW  = np.array(B[t_SW_start:t_SW_end])
Bx_SW = np.array(Bx[t_SW_start:t_SW_end])
By_SW = np.array(By[t_SW_start:t_SW_end])
Bz_SW = np.array(Bz[t_SW_start:t_SW_end])

T_SW, S_SW, PSD_tot_SW, PSD_para_SW, PSD_perp_SW = spectra_maker(t_SW, Bx_SW, By_SW, Bz_SW, 'linear')
B0_SW, Bx0_SW, By0_SW, Bz0_SW = fluct_maker(t_SW, Bx_SW, By_SW, Bz_SW, 'linear')[-4:]


B_Sh  = np.array(B[t_Sh_start:t_Sh_end])
Bx_Sh = np.array(Bx[t_Sh_start:t_Sh_end])
By_Sh = np.array(By[t_Sh_start:t_Sh_end])
Bz_Sh = np.array(Bz[t_Sh_start:t_Sh_end])

T_Sh, S_Sh, PSD_tot_Sh, PSD_para_Sh, PSD_perp_Sh = spectra_maker(t_Sh, Bx_Sh, By_Sh, Bz_Sh, 'linear')
B0_Sh, Bx0_Sh, By0_Sh, Bz0_Sh = fluct_maker(t_Sh, Bx_Sh, By_Sh, Bz_Sh, 'linear')[-4:]


B_MC  = np.array(B[t_MC_start:t_MC_end])
Bx_MC = np.array(Bx[t_MC_start:t_MC_end])
By_MC = np.array(By[t_MC_start:t_MC_end])
Bz_MC = np.array(Bz[t_MC_start:t_MC_end])

T_MC, S_MC, PSD_tot_MC, PSD_para_MC, PSD_perp_MC = spectra_maker(t_MC, Bx_MC, By_MC, Bz_MC, 'slide')
B0_MC, Bx0_MC, By0_MC, Bz0_MC = fluct_maker(t_MC, Bx_MC, By_MC, Bz_MC, 'slide')[-4:]


# In[ ]:


'''''''''''''''''''''''
   Plot preparation
'''''''''''''''''''''''

plt.close()

import matplotlib.gridspec as gridspec

def cm2inch(value):
    return value / 2.54

Page = plt.figure(figsize=(cm2inch(35), cm2inch(29.7)))
nb_lines = 10
nb_columns = 3
gs = gridspec.GridSpec(nb_lines, nb_columns)
fsize = 12

import matplotlib.colors       as colors

cmap = colors.LinearSegmentedColormap.from_list('nameofcolormap',
    ['rebeccapurple','b','c','g','forestgreen','gold','r','black'],gamma=0.9)

f_SW = 1.0 / S_SW[:,0]
f_Sh = 1.0 / S_Sh[:,0]
f_MC = 1.0 / S_MC[:,0]

# it should be that the lowest values are found in the solar wind
# and the highest in the sheath
PSD_min = np.min(PSD_tot_SW)/5
PSD_max = np.max(PSD_tot_Sh)*5

'''''''''''''''''''''''
    Temporal plots 
'''''''''''''''''''''''

'temporal plot of B'
axe = plt.subplot(gs[0:2,:])
axe.plot(time_vec, B, label="B")
axe.plot(t_SW, B0_SW, color='r')
axe.plot(t_Sh, B0_Sh, color='c')
axe.plot(t_MC, B0_MC, color='y')
axe.get_xaxis().set_visible(False)

axe.axvline(t_SW_start, color='r')
axe.axvline(t_SW_end, color='r')
axe.axvspan(t_SW_start, t_SW_end, color='r', alpha=0.2)
axe.axvline(t_Sh_start, color='c')
axe.axvline(t_Sh_end, color='c')
axe.axvspan(t_Sh_start, t_Sh_end, color='c', alpha=0.2)
axe.axvline(t_MC_start+n_avg, color='y')
axe.axvline(t_MC_end-n_avg, color='y')
axe.axvspan(t_MC_start+n_avg, t_MC_end-n_avg, color='y', alpha=0.2)

'temporal plot of Bx, By, Bz'
axe = plt.subplot(gs[2:4,:])
axe.plot(time_vec,Bx, label="Bx", color='blue')
axe.plot(time_vec,By, label="By", color='orange')
axe.plot(time_vec,Bz, label="Bz", color='green')

axe.plot(t_SW,Bx0_SW, color='r')
axe.plot(t_SW,By0_SW, color='r')
axe.plot(t_SW,Bz0_SW, color='r')

axe.plot(t_Sh,Bx0_Sh, color='c')
axe.plot(t_Sh,By0_Sh, color='c')
axe.plot(t_Sh,Bz0_Sh, color='c')

axe.plot(t_MC,Bx0_MC, color='y')
axe.plot(t_MC,By0_MC, color='y')
axe.plot(t_MC,Bz0_MC, color='y')

plt.legend()

Page.subplots_adjust(wspace=0)
Page.subplots_adjust(hspace=0)

'''''''''''''''''''''''
     Plot spectra
'''''''''''''''''''''''

f_min = 1 / n_avg
f_max = max(f_SW)

''' Plot Solar Wind Spectra '''

'Spectral signal B'
axe = plt.subplot(gs[5:,0])
axe.set_title(r'Solar Wind', weight = 'bold', fontsize = 16)

axe.scatter(f_SW,PSD_tot_SW      , label='total')
axe.scatter(f_SW,PSD_para_SW , label='parallel B0')
axe.scatter(f_SW,PSD_perp_SW , label='perpendicular B0')
axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlim(left=f_min)
axe.set_xlim(right=f_max)
axe.set_ylim([PSD_min, PSD_max])
axe.set_ylabel('Power $(nT^2.m^{-3})$')
plt.legend()

''' Plot Sheath Spectra '''

'Spectral signal B'
axe = plt.subplot(gs[5:,1])
axe.set_title(r'Sheath', weight = 'bold', fontsize = 16)

axe.scatter(f_Sh,PSD_tot_Sh      , label='total')
axe.scatter(f_Sh,PSD_para_Sh , label='parallel B0')
axe.scatter(f_Sh,PSD_perp_Sh , label='perpendicular B0')
axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlim(left=f_min)
axe.set_xlim(right=f_max)
axe.set_ylim([PSD_min, PSD_max])
axe.set_xlabel(r'$\Omega_{ci} / (2 \pi)$', weight='bold', fontsize=16)
axe.get_yaxis().set_visible(False)
plt.legend()

''' Plot Magnetic Cloud Spectra '''

'Spectral signal B'
axe = plt.subplot(gs[5:,2])
axe.set_title(r'Magnetic Cloud', weight = 'bold', fontsize = 16)

axe.scatter(f_MC,PSD_tot_MC  , label='total')
axe.scatter(f_MC,PSD_para_MC , label='parallel B0')
axe.scatter(f_MC,PSD_perp_MC , label='perpendicular B0')
axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlim(left=f_min)
axe.set_xlim(right=f_max)
axe.set_ylim([PSD_min, PSD_max])
axe.get_yaxis().set_visible(False)

plt.legend()

plt.show()


# In[ ]:


def rmsAoA(pos, Ax, Ay, Az, avg_type):
    dAx, dAy, dAz,_,_,_, A0,_,_,_ = fluct_maker(pos, Ax, Ay, Az, avg_type)
    
    rmsA = np.mean(np.sqrt(dAx**2 + dAy**2 + dAz**2))
    rmsAoA = rmsA / np.mean(A0)
    
    return rmsAoA


# In[ ]:


print(rmsAoA(t_SW, Bx_SW, By_SW, Bz_SW, 'linear'))
print(rmsAoA(t_Sh, Bx_Sh, By_Sh, Bz_Sh, 'linear'))
print(rmsAoA(t_MC, Bx_MC, By_MC, Bz_MC, 'slide'))


# print(rmsAoA(t_SW, Bx_SW, By_SW, Bz_SW, 'linear')) = 0.011777663289707661 
# 
# print(rmsAoA(t_SW, Bx_SW, By_SW, Bz_SW, 'slide'))  = 0.012127227586353205
# 
# Rassurant

# In[ ]:


# print(data['t00180']['rmsBoB_upstream'])
# print(data['t00220']['rmsBoB_upstream'])
# print(data['t00260']['rmsBoB_upstream'])


# In[ ]:


from IPython.core.display import Image, display


# In[ ]:


for time in data:
    plt.scatter(int(time[1:]), data[time]['rmsBoB_upstream'], color = 'blue')
    
plt.ylabel('rmsBoB_upstream')
plt.xlabel('time')
plt.show()


# rmsBoB calculé dans un cube n'est pas approprié hors régime stationnaire.
# 
# Tout simplement parce qu'une moyenne pour le cube par rapport à laquelle on calcule les fluctuations, donnera des fluctuations non nulles si la variation est linéaire.

# ### Small tests

# In[ ]:




