import netCDF4 as nc
import sys
import numpy as np
import pandas as pds

def readNetcdfFile3D(filepath,filename,str_data) :

    f = nc.Dataset(filepath+filename,'r')

    coord_sys = f.variables['Coordinate_system'][:].data

    coord_name = b''
    for i in range(0, len(coord_sys)):
        coord_name = coord_name + coord_sys[i]
    coord_name = str(coord_name)[2:].split(' ')[0]

    if coord_name=='simu':
        sgn = -1
    else:
        sgn = +1
   
    str_data.update({'c_omegapi': f.variables['phys_length'][:]})
    str_data.update({'x'        : np.flip(f.variables['X_axis'][:])})
    str_data.update({'y'        : np.flip(f.variables['Y_axis'][:])})
    str_data.update({'z'        : f.variables['Z_axis'][:]})

    str_data.update({'gstep' : f.variables['gstep'][:]})
 
    ### We use the array.transpose method for the following reason:
    ### arr = arr[z_axis,y_axis,x_axis]
    ### new_arr = arr.transpose(2, 1, 0)
    ### new_arr = new_arr[x_axis,y_axis,z_axis]

    if filename[0:4] == 'Magw' :
       
        Liste = ['Bx', 'By', 'Bz']
        for variable in Liste:
            f.variables[variable] = np.float32(f.variables[variable][:,:,:,])
        
        print('Reading Bx...')
        Bx = sgn*f.variables['Bx'][:,:,:].transpose(2, 1, 0)
        Bx = np.flip(Bx, axis=0)
        Bx = np.flip(Bx, axis=1)
        str_data.update({'Bx' : Bx})

        print('Reading By...')
        By = sgn*f.variables['By'][:,:,:].transpose(2, 1, 0)
        By = np.flip(By, axis=0)
        By = np.flip(By, axis=1)
        str_data.update({'By' : By})

        print('Reading Bz...')
        Bz = f.variables['Bz'][:,:,:].transpose(2, 1, 0)
        Bz = np.flip(Bz, axis=0)
        Bz = np.flip(Bz, axis=1)
        str_data.update({'Bz' : Bz})
   
    elif filename[0:3] == 'Hsw' :            
        Liste = ['Density', 'Ux', 'Uy', 'Uz',
                 'Temperature' ]
        for variable in Liste:
            f.variables[variable] = np.float32(f.variables[variable][:,:,:,])
       
        #print(filename) 
        print('Reading density...')
        N = f.variables['Density'][:,:,:].transpose(2, 1, 0)
        N = np.flip(N, axis=0)
        N = np.flip(N, axis=1)
        str_data.update({'n' : N})

        print('Reading Ux...')
        Vx = sgn*f.variables['Ux'][:,:,:].transpose(2, 1, 0)
        Vx = np.flip(Vx, axis=0)
        Vx = np.flip(Vx, axis=1)
        str_data.update({'Vx' : Vx})

        print('Reading Uy...')
        Vy = sgn*f.variables['Uy'][:,:,:].transpose(2, 1, 0)
        Vy = np.flip(Vy, axis=0)
        Vy = np.flip(Vy, axis=1)
        str_data.update({'Vy' : Vy})

        print('Reading Uz...')
        Vz = f.variables['Uz'][:,:,:].transpose(2, 1, 0)
        Vz = np.flip(Vz, axis=0)
        Vz = np.flip(Vz, axis=1)
        str_data.update({'Vz' : Vz})

        print('Reading T...')
        T = f.variables['Temperature'][:,:,:].transpose(2, 1, 0)
        T = np.flip(T, axis=0)
        T = np.flip(T, axis=1)
        str_data.update({'T' : T})

    elif filename[0:4] == 'Elew':
        
        Liste = ['Ex', 'Ey', 'Ez']
        for variable in Liste:
            f.variables[variable] = np.float32(f.variables[variable][:,:,:,])
        
        print('Reading Ex...')
        Ex = sgn*f.variables['Ex'][:,:,:].transpose(2, 1, 0)
        Ex = np.flip(Ex, axis=0)
        Ex = np.flip(Ex, axis=1)
        str_data.update({'Ex' : Ex})

        print('Reading Ey...')
        Ey = sgn*f.variables['Ey'][:,:,:].transpose(2, 1, 0)
        Ey = np.flip(Ey, axis=0)
        Ey = np.flip(Ey, axis=1)
        str_data.update({'Ey' : Ey})

        print('Reading Ez...')
        Ez = f.variables['Ez'][:,:,:].transpose(2, 1, 0)
        Ez = np.flip(Ez, axis=0)
        Ez = np.flip(Ez, axis=1)
        str_data.update({'Ez' : Ez})

    print('Close file and return...')
    f.close()
    return
    
def readNetcdf_grid(filepath,filename,str_data) :

    print(filepath+filename)
    f = nc.Dataset(filepath+filename,'r')


    str_data.update({'V_sw' : f.variables['vxs'][:]})
    str_data.update({'gstep' : f.variables['gstep'][:]})
    str_data.update({'pos_planet' : f.variables['s_centr'][:]})
    str_data.update({'c_omegapi' : f.variables['phys_length'][:]})
    str_data.update({'s_min' : f.variables['s_min'][:]})
    str_data.update({'s_max' : f.variables['s_max'][:]})
    
    str_data.update({'x ' : np.flip(f.variables['X_axis'][:])})
    str_data.update({'y ' : np.flip(f.variables['Y_axis'][:])})
    str_data.update({'z ' : f.variables['Z_axis'][:]})
    str_data.update({'ref_dens' : f.variables['phys_density'][:]})

    f.close()
