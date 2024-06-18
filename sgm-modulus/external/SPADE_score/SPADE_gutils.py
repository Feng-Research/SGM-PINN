import numpy as np
from scipy.sparse import csr_matrix, triu

def edgeList_adj(A):
    A_upper = triu(A, format='csr')
    edgeList = np.zeros((A_upper.nnz,2),dtype='int64')
    ##weightList = np.zeros((A_upper.nnz,1))
    idx = 0
    for i in range(0,A_upper.shape[0]):
        nodes = np.array([[i,j] for j in A_upper.indices[A_upper.indptr[i]:A_upper.indptr[i+1]]]).reshape(-1,2)
        ##weights = np.array(A_upper.data[A_upper.indptr[i]:A_upper.indptr[i+1]]).reshape(-1,1)
        edgeList[idx:idx+nodes.shape[0]] = nodes
        ##weightList[idx:idx+nodes.shape[0]] = weights
        idx = idx + nodes.shape[0]
    return edgeList##, weightList

def is_connected_adjacency(A): ##borrowing _plain_bfs from nx
    assert type(A) is csr_matrix, 'wrong matrix type'
    visited = set()
    nextlevel = {0} #start at 0
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in visited:
                visited.add(v)
                nextlevel.update(A.indices[A.indptr[v]:A.indptr[v+1]])
    return len(visited) == A.shape[0]

