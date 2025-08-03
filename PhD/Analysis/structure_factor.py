    # -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:06:43 2020

@author: Gil
"""

import matplotlib
import cupy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial import cKDTree
import psutil
import socket

def get_calculation_parameters(structure):
    """return the parameters that are necessary for the calculation
    N_points : The amount of points in the structure
    d_first_neighbours : The average distance between first neighbours"""
    
    N_points = len(structure)
    tree = cKDTree(structure)  # create tree of closest neighbours
    d, k = tree.query(structure,k=2)
    d_first_neighbours = np.average(d[:,1])
    
    return d_first_neighbours, N_points
    
def generate_vectors(structure, domain, vector_step, particule_distance): 
    """Generates the vector used for the calculation of the structure factor.
    distances_x, distances_y : vectors that contains the distances between all points along the x anf y axis respectivly
    scat_vector : the scatterring vector used for the calculation"""
    
    coord_x = structure[:,0]
    coord_y = structure[:,1]
   
    #create distance array for x an y   
    coord_x_ = coord_x.reshape(len(coord_x),1)
    distances_x = (coord_x - coord_x_).reshape(1,len(coord_x)**2)
        
    coord_y_ = coord_y.reshape(len(coord_y),1)
    distances_y= (coord_y - coord_y_).reshape(1,len(coord_y)**2)

    #set q vector
    border = domain/particule_distance;
    step = vector_step/particule_distance;
    scat_vector = np.arange(0,border,step)

    return distances_x, distances_y, scat_vector

def select_calculation_method(vector,q):
    """Select the calculation method depeding on memory of the computer, the maximum memory is half of the memory of the pc"""
    max_size_array = (2/3)*(psutil.virtual_memory().total/vector.itemsize)
    
    if max_size_array < np.size(vector,axis=1)*len(q):
       selector = "iterative"
       
    elif max_size_array > (np.size(vector,axis=1)*len(q)**2):
           selector = "matrix"
    else:
          selector = "matrix_by_parts"
          
    return selector

def iterative_method(d_x,d_y,q,N):
    
    structure_factor = np.zeros((len(q),len(q)))
    for a in range(len(q)):
        print(str(a/len(q)*100)+"%")
        for b in range(len(q)):
            structure_factor[a,b] = (1/N)*np.sum(np.exp(1j*(q[a]*d_x + q[b]*d_y)))
                
                
    return structure_factor
                
def generate_calculation_matrix(d_x,d_y,q):
    
    d_x = d_x.reshape(np.ma.size(d_x,axis=1),1)
    d_y = d_y.reshape(np.ma.size(d_y,axis=1),1)
                    
    mult_x = d_x*q
    mult_y = d_y*q
                    
    mult_y = mult_y.reshape(np.ma.size(mult_y,axis=0),np.ma.size(mult_y,axis=1),1)
    matrix_dq_y = np.swapaxes(mult_y,0,1)
                    
    mult_x = mult_x.reshape(np.ma.size(mult_x,axis=0),np.ma.size(mult_x,axis=1),1)
    mult_x = np.swapaxes(mult_x,0,1)
    matrix_dq_x = np.swapaxes(mult_x,0,2) 
    
    return matrix_dq_x, matrix_dq_y
        
def matrix_method(m_x,m_y,N,q_length):
    
    structure_factor = np.zeros((q_length,q_length))
    
    structure_factor =(1/N)*np.sum(np.cos(m_x+m_y), axis=1)
    
    return structure_factor

def matrix_by_parts_method(m_x,m_y,N,q_length):
    
    structure_factor = np.zeros((q_length,q_length))
    max_size_array = (2/3)*(psutil.virtual_memory().total/m_x.itemsize)
    max_dim = int(max_size_array / (np.ma.size(m_y,axis=1)*q_length))
    count = 1
    while np.ma.size(m_y,axis=0) > max_dim*count:
        structure_factor[(max_dim*(count-1)):max_dim*count,:] =(1/N)*np.sum(np.cos(m_x+m_y[(max_dim*(count-1)):max_dim*count,:]), axis=1)
        print(str((1-((np.ma.size(m_y,axis=0) - max_dim*count)/np.ma.size(m_y,axis=0)))*100) + "%")
        count+=1
                   
    structure_factor[(max_dim*(count-1)):,:] = (1/N)*np.sum(np.cos(m_x+m_y[(max_dim*(count-1)):,:]), axis=1)
    
    return structure_factor

def remove_center_2D(structure_factor_2D,qD, vector_end):
    
    x_q,y_q = np.meshgrid(qD,qD)
    X_r  = (x_q**2 + y_q**2)**0.5
    structure_factor_2D[( X_r < vector_end)] = np.min(structure_factor_2D)
    
    return structure_factor_2D 
   
def remove_center_1D(structure_factor_1D,qD):
    
   structure_factor_1D[(qD < 1)] = 0
    
   return structure_factor_1D

def remove_single_at_center(structure_factor):
    
    structure_factor[np.where(structure_factor == np.nanmax(structure_factor))] = 0

    return structure_factor    

def direct_Sq_calculation(structure,domain,vector_step, erase_center = False):

    """ Calculates the structure factor with the method given by the selector "method" """
    
    D, Ptot = get_calculation_parameters(structure)
    d_x, d_y, q = generate_vectors(structure, domain, vector_step, D)
    method = select_calculation_method(d_x,q)
    
    print(method)
    print(q[0]*1e6,q[-1]*1e6)
        
    if method == 'iterative':
      
        Sq_2D =  iterative_method(d_x,d_y,q,Ptot)
               
    elif method =='matrix':
        
        matrix_x,matrix_y = generate_calculation_matrix(d_x,d_y,q)
        Sq_2D = matrix_method(matrix_x,matrix_y,Ptot,len(q))
                   
    elif method == 'matrix_by_parts':
        
        matrix_x,matrix_y = generate_calculation_matrix(d_x,d_y,q)
        Sq_2D = matrix_by_parts_method(matrix_x,matrix_y,Ptot,len(q))
        
    if erase_center :
        Sq_2D = remove_center(Sq_2D,q,D)
        
    return Sq_2D, D, q

def radial_average(Sq,qD):
    
    N = len(Sq)
    #build matrix of radial distances
    x_q,y_q = np.meshgrid(qD,qD)
    R  = (x_q**2 + y_q**2)**0.5

    #array for radial bins
    step = qD[1]-qD[0]
    rad_bins = np.linspace(-step/2,np.max(qD)+step/2,num=N+1)

    #mid points for each bins to be used as x axis
    r= (rad_bins[0:-1]+rad_bins[1:])/2
    
    #calculate radial average
    structure_factor = np.zeros(N)

    for n in range(N-1):
        count = len(Sq[(R >= rad_bins[n]) & (R < rad_bins[n+1])])
        
        if count != 0:
    
            structure_factor[n]=np.sum(Sq[(R >= rad_bins[n]) & (R < rad_bins[n+1])]) / count
        
        else:
            structure_factor[n] = float('nan')
            
    return structure_factor,r

def save_data(Sq_2D, q, D, Sq, R, save_folder, filename):
    
    q_save = q.reshape(np.ma.size(q,axis=0),1)
    save_2D = np.append(Sq_2D, q_save,axis=1) 
    save_2D = np.append(save_2D, q_save*D,axis=1)
    
    save_1D = [Sq,R,R*D]

    np.savetxt(save_folder + '2D_Sq_' + filename + '_range_' + str(round((q[-1]+q[1]-q[0])*D)) 
               + '_step_' + str(round((q[1]-q[0])*D,3)) + '.dat', save_2D , delimiter = ',',header='The before last and last columns correspond to vector q and q.D respectively.')

    np.savetxt(save_folder + 'Sq_' + filename + '_range_' + str(round((q[-1]+q[1]-q[0])*D)) 
               + '_step_' + str(round((q[1]-q[0])*D,3)) +'.dat', save_1D, delimiter = ',',header='The row order is Sq,q,q.D.') 
    
def calculate_and_save(read_folder, file, domain, vector_step, default_folder = True, save_folder = ''):
    if default_folder:
        if socket.gethostname() == 'Gil-MSI':
            folder_write_files = 'D:/These/Programation/Structure_factor/Files/'

        elif socket.gethostname() == 'DESKTOP-JB8QHQB':
            folder_write_files = 'J:\Gil\Structure_factor/data_files/'
    
        else:
            folder_write_files = ''
            
    else:
        folder_write_files = save_folder
        
    print(file)    
    structure_to_calculate = np.genfromtxt(read_folder + file + '.csv', delimiter=',')
    Sq_2D, D, q = direct_Sq_calculation(structure_to_calculate,domain,vector_step)
    Sq,R = radial_average(Sq_2D,q)
    
    save_data(Sq_2D, q, D, Sq, R, folder_write_files, file)

def plot_Sq_2D (folder_read, file, edge, x_axis = 'qD', save_plot = False, remove_center = False, remove_single_center = False, 
                folder_write ='', log_scale = False, max_value = 100):

    Sq2D_and_arrays = np.loadtxt(folder_read + file + '.dat', delimiter=',' ) 

    Sq_2D = Sq2D_and_arrays[:,:-2]
    
    if x_axis == 'q':
        vector = Sq2D_and_arrays[:,-2]*1e6
        
    elif x_axis == 'qD':
        vector =   Sq2D_and_arrays[:,-1]
    
    elif x_axis == 'xf':
        vector =   Sq2D_and_arrays[:,-1]
        vector /= 1.033e2
        
    #Sq_2D = remove_center_2D(Sq_2D,vector,edge[0])
              
    bool_vec = (vector > -(edge[1])) & (vector < (edge[1]))
    vector = vector[bool_vec] 
    bool_vec_x,bool_vec_y = np.meshgrid(bool_vec,bool_vec)
    bool_vec_2D  = bool_vec_x & bool_vec_y    
    Sq_2D = Sq_2D[bool_vec_2D].reshape((len(vector),len(vector)))
    
    
    plt.rcParams.update({'font.size': 20})    
  
    #plot 2D structure factor and save figure
    figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.axes(xlim=(edge), ylim=(edge), autoscale_on=True)
    
    
    if log_scale:
        plot = plt.pcolor( vector , vector, Sq_2D, rasterized=True, cmap='jet', shading='auto', vmax= max_value, norm=matplotlib.colors.LogNorm())
        
    else :
        plot = plt.pcolor( vector , vector, Sq_2D, rasterized=True, cmap='jet', shading='auto', vmax= max_value)
    #plt.title('2D structure factor')
    
    if x_axis == 'q':
        plt.xlabel('qx (m\u207B\u00B9)')
        plt.ylabel('qy (m\u207B\u00B9)')
    
    elif x_axis == 'qD':
        plt.xlabel('qx (m\u207B\u00B9)')
        plt.ylabel('qy (m\u207B\u00B9)')
        
    elif x_axis == 'xf':
        plt.xlabel('x(micron)')
        plt.ylabel('y(micron)')
    
    plt.rc('axes', titlesize=20)     
    plt.rc('axes', labelsize=30) 
    plt.rc('xtick', labelsize=30)   
    plt.rc('ytick', labelsize=30)
    ax.set_aspect('equal')
    #plt.xlim(0, 4.5e7)   # set the xlim to left, right
    #plt.ylim(0, 4.5e7)
    cbar = plt.colorbar(plot, shrink = 0.65)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    
    if save_plot:
        
        if log_scale:  
            plt.savefig( folder_write + '2D_Sq_' + file + "_edge_" + str(edge) + '_log_scale.png')
        
        else:  
            plt.savefig( folder_write + '2D_Sq_' + file + "_edges_" + str(edge) + '.svg')
        
    plt.show() 
    
    
def vector_selection(array, axis_selection):

    if axis_selection == 'q':
        axis = array[-2]*1e6
    
    elif axis_selection == 'qD':
        axis = array[-1]
        
    elif axis_selection == 'xf':    
        axis = array[1]*1e6
        axis /= 1.033e4     
        
    return axis

    
def plot_Sq_1D(folder_read, file, labels, edges, x_axis = 'qD', save_plot = False, folder_write ='', log_scale = False, mult_plot=False, moving_average = False, n_ave = 3, y_max = 100):
    
    figure(num=None, figsize=(18, 10), dpi=100, facecolor='w', edgecolor='k')
    
    markers_array = np.asarray(["v", "o", "^", "s", "<", "x", ">","v", "o", "^", "s", "<", "x", ">"])
    for count, Sq_and_array in enumerate(file):
        
        Sq_and_arrays = np.genfromtxt(folder_read + str(file[count]) + '.dat', delimiter=',')
        
        Sq = Sq_and_arrays[0]
        
        
        if moving_average:
            
            Sq_mov_ave = np.zeros(len(Sq) - n_ave)
            
            for p in range(n_ave):
                print(p)
                Sq_mov_ave += Sq[p:p - n_ave] 
            
            Sq_mov_ave /= n_ave
            Sq = np.append(Sq[:n_ave],Sq_mov_ave)
            
        vector = vector_selection(Sq_and_arrays,x_axis)
                
        Sq = Sq[(np.logical_and((edges[0] < vector), (vector < edges[1])))]
        vector = vector[(np.logical_and((edges[0] < vector), (vector < edges[1])))] 


        plt.plot(vector,Sq, marker = markers_array[count], ms=6, lw=2, label= labels[count])
        plt.rc('legend', fontsize=15)
        plt.rc('axes', titlesize=30)     
        plt.rc('axes', labelsize=30) 
        plt.rc('xtick', labelsize=30)   
        plt.rc('ytick', labelsize=30) 
        
    if log_scale: 
        plt.yscale("log")
    
    if x_axis == 'q':
        plt.xlabel('q (m\u207B\u00B9)')
    
    elif x_axis == 'qD':
        plt.xlabel('q (m\u207B\u00B9)')
        
    elif x_axis == 'xf':
        plt.xlabel('x(micron)')
    
    plt.ylabel('S(q)')
    plt.axvline(x=2.19e6, color='k', lw = 2, linestyle='dashed')
    plt.axvline(x=2.22e7, color='k', lw = 2, linestyle='dashed')
    plt.axvline(x=1.57e7, color='k', lw = 2, linestyle='dashed')
    plt.axhline(y=0.05, color='g', lw = 2)
    plt.axhline(y=0.1, color='b', lw = 2)
    plt.ylim([0,y_max])
    plt.xlim(edges)
    plt.legend()
    plt.tight_layout
    
    if save_plot:
        plt.tight_layout
        plt.savefig( folder_write + 'Sq_' + 'multiple_filtered_areas' + "_edges_" + str(edges[0]) + '_' + str(edges[1]) + '.svg')