def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#... import sklearn stuff...
import csv
import copy
import gc
import time
import psutil
import math
import hnswlib
import numpy as np
import pickle
from sklearn import preprocessing
from scipy.sparse import csr_matrix, csc_matrix, diags, identity, triu, tril
import networkx as nx
from functools import reduce
import shutil

def loadData_true(npzfile, clean = False):
    start = time.time()
    load = np.load(npzfile, allow_pickle=True)
    lst = load.files
    refDict = load[lst[0]][()]
    refKeys = list(refDict.keys())
    partLen = len(refDict[refKeys[0]])
    parts = len(lst)
    out = {i:np.zeros((partLen*parts,1)) for i in refKeys}
    for k,i in enumerate(lst):
        temp = load[i][()]
        for j in refKeys:
            out[j][k*partLen:(k+1)*partLen] = temp[j]
    if clean:
        cut = -1
        for i in refKeys:
            out[i] = cleanData(out[i])
            if cut < 0:
                cut = len(out[i])
            cut = min(cut,len(out[i]))
        for i in refKeys:
            out[i] = out[i][:cut]
    end = time.time()
    print(end-start)
    return out

def loadData(npzfile, exclude_list= []):
    exclude = [i.lower() for i in exclude_list]
    load = np.load(npzfile, allow_pickle=True)
    lst = load.files
    rdict = load[lst[0]][()]
    #lkeys = [i for i in rdict.keys() if all([(j != i.lower()) for j in exclude])]
    lkeys = [i for i in rdict.keys() if all([((j != i.lower()) and 
        #not(j=='lambda*' and j[:-1] in i.lower())) for j in exclude])]
        not('*' in j.lower() and j[:-1] in i.lower())) for j in exclude])]
    nData = np.concatenate(([rdict[i] for i in lkeys]),axis=1)
    dim = nData.shape[1]
    data = nData
    for i in lst[1:]:
        rdict = load[i][()]
        nData = np.concatenate(([rdict[i] for i in lkeys]),axis=1)
        #data[(k+1)*b:(k+1)*b+b,:] = nData
        data = np.concatenate((data,nData),axis=0)
    data = cleanData(data)
    num_elements = data.shape[0]
    ids = np.arange(num_elements)
    return data, ids, lkeys, num_elements, dim

def simnetLoadData(npzfile):
    start = time.time()
    load = np.load(npzfile, allow_pickle=True)
    nparray = load[load.files[0]][()]
    end = time.time()
    print(end-start)
    return nparray

def getMappingCore(P, total = 1, repetition = True, shuffle = False, sample = 1):
    if repetition == True:
        return getMappingCore_repeats_shuffle_ratio(P,total = total, shuffle = shuffle, sample = sample)
    choices = getMappingCore_repeats_shuffle_ratio(P, total = 1, shuffle = False, sample = sample)
    print(f'UNCONTROLLED epochs from {sample} points selected from each cluster.')
    coreLength = total*P.shape[1]
    try:
        rand = np.random.default_rng()
        idxs = np.rand.choice(choices.shape[0],coreLength)
    except:
        idxs = np.random.choice(choices.shape[0],coreLength)
    out = np.array([choices[i] for i in idxs])
    return out

def getMappingCore_repeats_shuffle_ratio(P, total = 1, shuffle = False, sample = 1): 
    #optionally shuffles output epochs, 
    #total is the # of times to repeat the output in a single array, 
    #sample is the # of samples to get from each cluster. Takes an absolute # of samples (up to cluster size) if sample >1, or a percentage of samples in each cluster if 0<sample<1
    if not type(P) == csc_matrix:
        P = P.tocsc(copy=False)
    n = P.shape[1]
    print(f'Clusters: {n}, Total: {n*total}, Repeats: {total}, shuffle: {shuffle}')
    start = time.time()
    #columns is an array of the number of elements in each column (samples per cluster)
    columns = np.array([P.indptr[i+1]-P.indptr[i] for i in range(0,P.shape[1])])
    #selections is an array of index choices for each row 
    if sample > 1:
        selectSize = (lambda x : min(x,sample))
    elif 0 < sample < 1:
        print(sample)
        selectSize = (lambda x : math.ceil(x*sample))
    else:
        selectSize = (lambda _: (1,))
    try:
        rand = np.random.default_rng() #for later versions of numpy
        selections = [rand.choice(i, size = selectSize(i), replace = False) for i in columns]
    except:
        selections = [np.random.choice(i,size = selectSize(i), replace = False) for i in columns] #for earlier versions of numpy
    out = np.hstack([ np.array([P.indices[P.indptr[i]:P.indptr[i+1]][np.array(selections[i])], #the indexes of samples
                          np.array([columns[i] for j in selections[i]]), #the size of the cluster
                          np.array([i for j in selections[i]]), #the absolute number (ID) of the cluster
                          np.array([selections[i].shape[0] for j in selections[i]]) #the number of samples in the cluster
                        ]) for i in range(0,n)])
    out = out.T
    if sample != 1:
        n = out.shape[0]
    print(n)
    if total > 1:
        out = np.tile(out,(total,1))
    print(out.shape)
    end = time.time()
    print(end-start)
    if shuffle:
      for i in range(0,out.shape[0],n):
          if i == 0:
              print(f'MappingShuffle {i},{i+n}')
          np.random.shuffle(out[i:i+n,:]) #shuffles along axis 1
    return out

def getMappingCore_weighted_fast_modulus(P, cluster_subset, cluster_values, sMin = .1, sMax = .8, inPath = '', coarsePath = '', total = 1, shuffle = False, avg = False):
    '''Training outputs only, not set up to combine with inference data...'''
    assert sMin > 0 and sMax < 1 and sMin < sMax

    print(f'data processed here {cluster_values.shape}')
    cluster_values = np.abs(cluster_values) #ensure positive, they should all be losses but...
    cluster_values = preprocessing.scale(cluster_values)
    cluster_values = np.sum(cluster_values, axis=1) #sum losses of each sample
    print(f'data processed here {cluster_values.shape}')
    print(cluster_subset.shape)
    assert cluster_values.shape[0] == cluster_subset.shape[0]
    
    #map cluster subsampled losses back to their clusters 
    idx = 0
    step = 0
    stop = cluster_values.shape[0] #total number of subsamples
    n_data = np.zeros((P.shape[1])) #final number of clusters
    while idx < stop:
        stepSize = cluster_subset[idx][3] #steps to the beginning of the next cluster
        n_data[step] = np.sum(cluster_values[idx:idx+stepSize]) #summing values from this step's cluster
        if avg:
            n_data[step] = n_data[step]/cluster_subset[idx][1]
        idx = idx + stepSize #move idx to next cluster
        step = step + 1 #iterate step
    assert n_data.shape[0] == P.shape[1]

    if not type(P) == csc_matrix:
        P = P.tocsc(copy=False)
    n = P.shape[1]

    print(f'Clusters: {n}, Total: {n*total}, Repeats: {total}, shuffle: {shuffle}')
    start = time.time()
    #columns is an array of the number of elements in each column (samples per cluster)
    columns = np.array([P.indptr[i+1]-P.indptr[i] for i in range(0,n)])
    #clusterW is an array of the sums of the features of all elements in each cluster
    #clusterW = np.array([np.sum(data[P.indices[P.indptr[i]:P.indptr[i+1]]]) for i in range(0,n)])
    q = np.percentile(n_data, [98])
    limit = q*2
    clusterW = np.where(n_data >= limit, limit, n_data)
    clusterW = preprocessing.minmax_scale(clusterW, feature_range = (sMin,sMax), axis=0)
    print(f'ClusterW: {clusterW.shape}')
    assert columns.shape == clusterW.shape
    #selectSize is the number of samples to collect from each cluster
    selectSize = (np.ceil(clusterW*columns)).astype(int)
    #selections is an array of arrays of index choices for each column/cluster
    try:
        rand = np.random.default_rng() #for later versions of numpy
        selections = [rand.choice(columns[i], size = selectSize[i], replace = False) for i in range(0,n)]
    except:
        selections = [np.random.choice(columns[i], size = selectSize[i], replace = False) for i in range(0,n)] #for earlier versions of numpy
    out = np.hstack([ np.array([P.indices[P.indptr[i]:P.indptr[i+1]][np.array(selections[i])], #the indexes of samples
                        np.array([columns[i] for _ in selections[i]]), #the size of the cluster
                        np.array([i for _ in selections[i]]), #the absolute number (ID) of the cluster
                        np.array([selections[i].shape[0] for _ in selections[i]]), #the number of samples selected from the cluster
                        np.array([clusterW[i] for _ in selections[i]])
                    ]) for i in range(0,n)])
    out = out.T 
    n = out.shape[0]
    print(n)
    if total > 1:
        out = np.tile(out,(total,1))
    print(out.shape)
    if shuffle:
        for i in range(0,out.shape[0],n):
            if i == 0:
                print(f'MappingShuffle {i},{i+n}')
            np.random.shuffle(out[i:i+n,:]) #shuffles along axis 1
    end = time.time()
    print(end-start)
    # #saving for analysis
    # print('Saving: ',BC)
    # np.savez(coarsePath + BC +'_weighted.npz', out)
    return out

def HNSWtoG_noData(labels,distances, 
        reciprocal = True, scale = True, square = True,
        featMin = 1, featMax = 10, offset = 0):
    start = time.time()
    wd = distances[:,1::]
    if square:
        print('sq')
        wd = np.square(wd)
    if reciprocal:
        print('rec')
        wd = np.reciprocal(wd)
    if scale:
        print('prep')
        wd = preprocessing.minmax_scale(wd, feature_range = (featMin,featMax), axis=1)
    d = {node+offset: {labels[node,i+1]+offset: {"weight": wd[node,i]} for i in range(0,wd.shape[1])} for node in range(0,labels.shape[0])}
    end = time.time()
    print(end-start)
    return nx.from_dict_of_dicts(d)

def cleanData(data):
    hold = data[-1,:]
    for k,i in enumerate(data[::-1,:]):
        if not (abs(i-hold) < 1e-6).all():
            print(str(k) + ',' + str(len(data)) + ',' + str(len(data)-k+1))
            return data[:len(data)-k+1,:]
    return data

def cleanData_cols(data): #vertical
    hold = data[-1,:]
    cut = data.shape[0]
    for column in range(0,data.shape[1]):
        for k,i in enumerate(data[::-1,column]):
            if not(abs(i-hold[column] < 1e-6)):
                print(str(k) + ',' + str(len(data)) + ',' + str(len(data)-k+1))
                cut = min(data.shape[0]-k+1,cut)
                break
    return data[:cut,:]