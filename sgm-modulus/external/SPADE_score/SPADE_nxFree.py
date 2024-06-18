import hnswlib
import numpy as np
from scipy.sparse import csr_matrix
from julia.api import Julia
from scipy.sparse.csgraph import laplacian
import modulus.external.SPADE_score.SPADE_gutils as sgu

from sklearn import preprocessing
import os 

def spade(data_input, data_output, k=10, num_eigs=2,
          sparse=False,weighted=False,
          wID = -1, backups = 4):

    print(f'{wID} SPADE in K: {k}, NUM EIGS: {num_eigs}')
    neighs_in, distance_in = hnsw(data_input, k)
    

    #if G is not connected
    backups = 4
    backup_k = [k*(2**i) for i in range(1,backups)]
    counter=0
    while True:
        if weighted:
            adj_in, _, edges_in = construct_weighted_adj(neighs_in, distance_in)
        else:
            adj_in, _, edges_in = construct_adj(neighs_in, distance_in)
        try:
            print(f'{wID} input checking connectedness')
            assert sgu.is_connected_adjacency(adj_in), "input graph is not connected"
            print(f'{wID} input connected!')
            break
        except AssertionError:
            np.savez(f'{wID}_inputLabelsDists_{counter}.npz',neighs_in, distance_in,data_input)
            print(f'{wID} input Trying K={backup_k[counter]}, ef={200+backup_k[counter]}, out of {backup_k}') 
            neighs_in, distance_in = hnsw(data_input, backup_k[counter], 200+backup_k[counter])
            counter += 1
            continue
        
    print(f'{wID} SPADE out K: {k}, NUM EIGS: {num_eigs}')
    neighs_out, distance_out = hnsw(data_output, k)
    counter=0
    while True:
        if weighted:
            adj_out, _, edges_out = construct_weighted_adj(neighs_out, distance_out)
        else:
            adj_out, _, edges_out = construct_adj(neighs_out, distance_out)
        try:
            print(f'{wID} output checking connectedness')
            assert sgu.is_connected_adjacency(adj_out), "output graph is not connected"
            print(f'{wID} output connected!')
            break
        except:
            np.savez(f'{wID}_outputLabelsDists_{counter}.npz',neighs_out,distance_out,data_output)
            print(f'{wID} ouput Trying K={backup_k[counter]}, ef={backup_k[counter]*2}, out of {backup_k}')
            neighs_out, distance_out = hnsw(data_output, backup_k[counter], 200+backup_k[counter])
            counter += 1
            continue
        
    if sparse:
        adj_in = SPF(adj_in, 4)
        adj_out = SPF(adj_out, 4)
    L_in = laplacian(adj_in, normed=False)
    L_out = laplacian(adj_out, normed=False)
  
    TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy = GetRiemannianDist(edges_in, edges_out, L_in, L_out, num_eigs)# full function
    return TopEig, TopEdgeList, TopNodeList, node_score, L_in, L_out, Dxy, Uxy


def hnsw(features, k=10, ef=200, M=24):
    print(f'HNSW K: {k}, EF: {ef}, M: {M}')
    num_samples, dim = features.shape
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, distance = p.knn_query(features, k+1)
  
    return neighs, distance


def construct_adj(neighs, weight):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    adj.data[:] = 1
    lap = laplacian(adj, normed=False)
    # construct a edgelist from the adjacency matrix
    edges = sgu.edgeList_adj(adj)
    return adj, lap, edges

def construct_weighted_adj(neighs, distances):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    weights = np.exp(-distances)

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    # calculate weights for each edge
    edge_weights = weights[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    all_data = np.concatenate((edge_weights, edge_weights), axis=0)  # use weights instead of ones
    adj = csr_matrix((all_data, (all_row, all_col)), shape=(dim, dim))
    lap = laplacian(adj, normed=False)
    # construct a edgelist from the adjacency matrix
    edges = sgu.edgeList_adj(adj)
    return adj, lap, edges, edge_weights

def SPF(adj, L, ICr=0.11):
    jl = Julia(compiled_modules=False)
    from julia import Main
    ##Main.include("./my_utils/SPF.jl")
    Main.include("/modulus/modulus/external/SPADE_score/SPF.jl")
    agj_c = Main.SPF(adj, L, ICr)

    return agj_c

def GetRiemannianDist(edges_in, edges_out, Lx, Ly, num_eigs): 
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()
    num_edge_tot=edges_in.shape[0] # number of total edges  
    Zpq=np.zeros((num_edge_tot,));# edge embedding distance
    p = edges_in[:,0];# one end node of each edge
    q = edges_in[:,1];# another end node of each edge
    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    TopNodeList = np.flip(node_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy

def julia_eigs(l_in, l_out, num_eigs):
    jl = Julia(compiled_modules=False)
    from julia import Main
    ##Main.include("./my_utils/eigen.jl")
    Main.include("/modulus/modulus/external/SPADE_score/eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, num_eigs)
    return eigenvalues.real, eigenvectors.real






