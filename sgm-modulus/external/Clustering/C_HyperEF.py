import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import numpy as np
import os 
import time
from scipy.sparse import csc_matrix
import gc
import networkx as nx
from scipy.sparse import lil_matrix
from sklearn import preprocessing
import hnswlib
import numpy as np

print(f"julia_main", flush = True)
def hnsw(data, k):


    dim = data.shape[1]
    num_elements = data.shape[0]
    data_labels = np.arange(num_elements)

    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k)

    return labels, distances

class C_HyperEF:

    def __init__(self):
        
        from julia.api import Julia
        print("Imported Julia")
        jl = Julia(compiled_modules=False)
        from julia import Main
        Main.include("/modulus/modulus/external/Clustering/HyperEF/HyperEF.jl")
        print("Imported Main")
        self.JLMainGC = Main.GC.gc()
        self.HyperEF_fcn = Main.HyperEF

    @staticmethod
    def dict_to_numpy(invar):
        data = np.concatenate([invar[x] for x in invar.keys()], axis=1)
        num_elements = list(invar.values())[0].shape[0]
        ids = np.arange(num_elements)
        lkeys = invar.keys()
        dim = len(lkeys)
        return data, ids, lkeys, num_elements, dim

    def HyperEF_P(self, invar, initial_vars, k, L): 
        invar = {k:v for k,v in invar.items() if k in initial_vars}
        print(f'Using: {str(initial_vars)}, KNN: {k}, Level: {L}')
        data, ids, lkeys, num_elements, dim = self.dict_to_numpy(invar)
        print('Keys:' + str(lkeys))
        print(f'Size:({num_elements}x{dim})')
        print('Norm')
        data_output = preprocessing.scale(data)
        I_out, w_out = hnsw(data_output, k)
        A_out = lil_matrix((data_output.shape[0], data_output.shape[0]))

        for ii in range(0, data_output.shape[0]):
            iii = 1
            for j_out in I_out[ii][1:]:
                A_out[ii,j_out] = w_out[ii,iii]
                iii += 1
        try:
            G = nx.from_scipy_sparse_matrix(A_out)
        except:
            G = nx.from_scipy_sparse_array(A_out)
        g_list = list(G.edges())
        for i in range(len(g_list)):
            #g_list[i] = list(g_list[i])
            g_list[i] = np.array(g_list[i], dtype='int64')
            for j in range(len(g_list[i])):
                g_list[i][j] += 1
        g_list = tuple(g_list)
        #g_list = np.array(g_list, dtype='object')
        idx  = self.HyperEF_fcn( g_list, L)
        print("IDX Done")
        row = list(range(len(idx)))
        col = [i-1 for i in idx]
        print(f"idxs shifted")
        self.JLMainGC
        print(f'JuliaGC:')
        P = csc_matrix((list(range(len(idx))),(row,col)), shape = (data_output.shape[0],np.unique(idx).size))
        return P


if __name__ == '__main__':
    #main()
    pass
