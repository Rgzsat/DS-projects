from __future__ import print_function
import matplotlib.pyplot as plt
import pywt
import numpy as np
import random
from math import *

#%%

def data_scale(r_data):
    dim = r_data.shape[1]  
    # the data has at least 2 raw, label and signal
    min_list = [np.amin(r_data[:,i]) for i in range(0, dim)]
    max_list = [np.amax(r_data[:,i])+0.001 for i in range(0, dim)] 
    # add the [0] and [1] because there is a 'raw of lable', and 0.001 to avoid 1
    toZero = r_data - np.array(min_list)
    norm_data = toZero / (np.array(max_list) - np.array(min_list))
    return(norm_data)

def map_sc_domain(data, scale=124):
    # it applies the wavelet transform so as to map the data
    if scale <= 0 or not(isinstance(scale, int)):
        raise ValueError('Error, "scale" is a positive integer ')
    length = data.shape[0]
    dim = data.shape[1]
    data_sd = {}
    for i in range(0,length):
        num = 0
        for j in reversed(range(0, dim)):     # initialize in the most weighted dimension
            num=num+(data[i,j]//(1/scale))*pow(scale, j)  # it starts from 0
        num = int(num)
        if data_sd.get(num, 'N/A')=='N/A':
            data_sd[num] = 1
        else: data_sd[num]=data_sd[num]+1
    return(data_sd)

#%%
def WT_nd(data, dim, scale, wave):
    # it is in the 1 order and "n" dimension, wavelet transform with numbered grids
    wave_let = {'db1':[0.707, 0.707], 'bior1.3':[-0.09, 0.09, 0.707, 0.707, 0.09, -0.09], \
                'db2':[-0.13, 0.224, 0.836, 0.483]}
    low_freq = {}
    len_convolution = len(wave_let.get(wave))-1
    len_line = ceil(scale/2) + ceil((len_convolution-2)/2)
    for dim_in in range(0, dim):
        for key in data.keys():
            coord = [] # coord start from 0
            tempkey = key
            for i in range(0,dim):
                # get the coord for a numbered grid
                if i <= dim-dim_in-1:
                    coord.append(tempkey//pow(scale, (dim-1-i)))
                    tempkey = tempkey%pow(scale, (dim-1-i))
                else:
                    coord.append(tempkey//pow(len_line, (dim-1-i)))
                    tempkey = tempkey%pow(len_line, (dim-1-i))
            coord.reverse()
            in_coord = ceil((coord[dim_in]+1)/2)-1    # to calculate WT_nd, signal should start from 1
            in_num = 0    # numbered lable for next level of data
            for i in range(0, dim):
                if i <= dim_in:
                    if i == dim_in:
                        in_num += in_coord*pow(len_line, i)
                    else:
                        in_num += coord[i]*pow(len_line, i)
                else:
                    in_num += coord[i]*pow(scale, i)
            wavelet = wave_let.get(wave)   # to perform convolution
            for i in range(0, len_convolution//2+1):  
                if in_coord+i >= len_line: # coord start from 0 
                    break
                if low_freq.get(int(in_num+pow(len_line, dim_in)*i), 'N/A') == 'N/A':
                    low_freq[int(in_num+pow(len_line, dim_in)*i)] = \
                            data[key]*wavelet[int((in_coord+1+i)*2-(coord[dim_in]+1))]
                else:
                    low_freq[int(in_num+pow(len_line, dim_in)*i)] += \
                            data[key]*wavelet[int((in_coord+1+i)*2-(coord[dim_in]+1))]
        data = low_freq
        low_freq = {}
    return(data)

#%%

class find_node():
    def __init__(self,key=0,value=0):
        self.cluster = None
        self.key = key
        self.value = value
        self.process = False
    def around(self,scale=1,dim=1):
        node_key_around = []
        coordinate = []
        for ini_dim in range(0,dim):
            coord_dim = self.key//pow(scale,ini_dim)
            if coord_dim == 0:
                node_key_around.append(self.key+pow(scale,ini_dim))
            elif coord_dim == scale-1:
                node_key_around.append(self.key-pow(scale,ini_dim))
            else:
                node_key_around.append(self.key+pow(scale,ini_dim))
                node_key_around.append(self.key-pow(scale,ini_dim))
        return(node_key_around)
    
#%%

def bfs(pair_equal,queue_max):
    if pair_equal == []:
        return(pair_equal)
    grou = {x:[] for x in range(1,queue_max)}
    result = []
    for i,j in pair_equal:
        grou[i].append(j)
        grou[j].append(i)
    for k in range(1,queue_max):
        if k in grou:
            if grou[k] == []:
                del grou[k]
            else:
                queue = [k]
                for l in queue:
                    if l in grou:
                        queue= queue+ grou[l]
                        del grou[l]
                record = list(set(queue))
                record.sort()
                result.append(record)
    return(result)

def key_cluster_build(nodes,list_equal,min_cluster_cut):
    key_cluster = {}
    for point in nodes.values():
        flag = 0
        for cluster in list_equal:
            if point.cluster in cluster:
                point.cluster = cluster[0]
                if key_cluster.get(cluster[0],'N/A') == 'N/A':
                    key_cluster[cluster[0]] = [point]
                    flag = 1
                else:
                    key_cluster[cluster[0]].append(point)
                    flag = 1
                break
        if flag == 0:
            if key_cluster.get(point.cluster,'N/A') == 'N/A':
                key_cluster[point.cluster] = [point]
            else:
                key_cluster[point.cluster].append(point)
    count = 1
    res = {}
    for cluster in key_cluster.keys():
        if len(key_cluster[cluster]) == 1:
            if key_cluster[cluster][0].value < min_cluster_cut:
                continue
        for p in key_cluster[cluster]:
            res[p.key] = count
        count= count+1
    return(res)

#%%
def calc_cluste(data, scale, dim, min_cluster_cut):
    "i= denotes point"
    "j= denotes around"
    pair_equal = []
    flag_cluster = 1
    for i in data.values():
        i.process = True
        for j in i.around(scale, dim):
            if not(data.get(j, 'N/A')=='N/A'):
                j = data.get(j)
                if j.cluster is not None:
                    if i.cluster is None:
                        i.cluster = j.cluster
                    elif i.cluster != j.cluster:
                        mincluster = min(i.cluster, j.cluster)
                        maxcluster = max(i.cluster, j.cluster)
                        pair_equal =pair_equal+[(mincluster,maxcluster)]
        if i.cluster is None:
            i.cluster = flag_cluster
            flag_cluster= flag_cluster+ 1
    pair_equal = set(pair_equal)
    equal_list = bfs(pair_equal,flag_cluster)
    result = key_cluster_build(data,equal_list,min_cluster_cut)
    return(result)

def pro_threshold(data,threshold,scale,dim):
    "i=key"
    "j=value"
    nodes = {}
    result = {}
    ini_node = find_node(0)
    avg = 0
    for i,j in data.items():
        if j >= threshold:
            nodes[i]=find_node(i,j)
            avg=avg+ j
            if j > ini_node.value:
                ini_node = find_node(i,j)
    min_cluster_cut = avg/len(nodes)
    clusters = calc_cluste(nodes,scale,dim,min_cluster_cut)
    return(clusters)

#%%

#Function to calculate the threshold

def f_Threshold(data,threshold):
    "i=point"
    val = list(data.values())
    val.sort(reverse=True)
    d1 = [i for i in range(1,len(val)+1)]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(d1,val)
    ax.axhline(y=threshold,xmin=0,xmax=1,color='r')
    plt.show()
    
def data_mark(data_norm,cluster,scale):
    "i=point"
    "column=tag"
    dim = data_norm.shape[1]
    # it represents a column for tags
    cols = []
    for i in range(0,data_norm.shape[0]):
        number = 0
        for in_dim in range(0,dim):
            number=number+ (data_norm[i,in_dim]//(1/scale))*pow(scale,in_dim)
        if cluster.get(int(number),'N/A')=='N/A':
            cols.append(0)
        else:
            cols.append(cluster.get(int(number)))
    return(cols)

#%%
#FINAL STEPS

def calc_wav_cluste(data,scale=40,wavelet='db2',threshold=0.35,plot=False):
    # 
    waveletlen = {'db1':0,'db2':1,'bior1.3':2}
    data_norm = data_scale(data)
    dim = data_norm.shape[1]
    dic_data = map_sc_domain(data_norm,scale)
    dwt_resu = WT_nd(dic_data,dim,scale,wavelet)
    if plot: f_Threshold(dwt_resu,threshold) #to calculate the treshold 
    lineLen = scale//2+waveletlen.get(wavelet)
    result = pro_threshold(dwt_resu,threshold,lineLen,dim)
    tags = data_mark(data_norm,result,lineLen)
    return(tags)

#%%

from pyclustering.samples.definitions import FCPS_SAMPLES
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from pyclustering.utils import read_sample


df = np.array(read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS))
 #Example to plot the threshold
wav_c1 = calc_wav_cluste(df, scale=40, threshold=0.30, plot=True) #implementation of wavecluster
true_c1 = np.arange(len(df))>=400

#%%Final plots
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
color = wav_c1 / np.amax(wav_c1)
rgb = plt.get_cmap('jet')(color)
ax.scatter(df[:,0],df[:,1],color = rgb) 
plt.show()
#%%

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
color = true_c1 / np.amax(true_c1)
rgb = plt.get_cmap('jet')(color)
ax.scatter(df[:,0],df[:,1],color = rgb) 
plt.show()
