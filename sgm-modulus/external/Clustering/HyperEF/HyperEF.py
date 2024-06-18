julia_main = "/modulus/modulus/external/Clustering/HyperEF/HyperEF.jl"

print(f"julia_main", flush = True)
def hnsw(data, k):
    import hnswlib
    import numpy as np
    

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

def HyperEF(data_output, k, L):
    import numpy as np
    import networkx as nx
    from scipy.sparse import lil_matrix

    I_out, w_out = hnsw(data_output, k)      
    A_out = lil_matrix((data_output.shape[0], data_output.shape[0]))
    
    for ii in range(0, data_output.shape[0]):
        iii = 1
        for j_out in I_out[ii][1:]:
            A_out[ii,j_out] = w_out[ii,iii]
            iii += 1

    G = nx.from_scipy_sparse_matrix(A_out)
    g_list = list(G.edges())
    for i in range(len(g_list)):
        g_list[i] = np.array(g_list[i], dtype='int64')
        for j in range(len(g_list[i])):
            g_list[i][j] += 1
    g_list = tuple(g_list)
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./HyperEF.jl")
    idx  = Main.HyperEF( g_list, L)
    selected_id = []
    for i in range(max(idx)):
        ids = list(np.where(idx == i+1)[0])
        selected_id.append(np.random.choice(ids))
    return selected_id


def HyperEF_P(data_output, k, L): #small edit to output as a sparse matrix similar to sim_coarse
    import numpy as np
    import networkx as nx
    from scipy.sparse import lil_matrix
    from scipy.sparse import csc_matrix

    I_out, w_out = hnsw(data_output, k)      
    A_out = lil_matrix((data_output.shape[0], data_output.shape[0]))
    
    for ii in range(0, data_output.shape[0]):
        iii = 1
        for j_out in I_out[ii][1:]:
            A_out[ii,j_out] = w_out[ii,iii]
            iii += 1

    G = nx.from_scipy_sparse_matrix(A_out)
    g_list = list(G.edges())
    for i in range(len(g_list)):
        g_list[i] = np.array(g_list[i], dtype='int64') 
        for j in range(len(g_list[i])):
            g_list[i][j] += 1
    g_list = tuple(g_list)
    
    from julia.api import Julia
    print("Imported Julia")
    jl = Julia(compiled_modules=False)
    from julia import Main
    print("Imported Main")
    Main.include("/modulus/modulus/external/HyperEF/HyperEF.jl")
    Main.include(julia_main)

    print(f"Starting HyperEF at {julia_main}")
    idx  = Main.HyperEF( g_list, L)
    print("IDX Done")
    row = list(range(len(idx)))
    col = [i-1 for i in idx]
    print(f"idxs shifted")
    P = csc_matrix((list(range(len(idx))),(row,col)), shape = (data_output.shape[0],np.unique(idx).size))
    print(f"P created")
    return P


def contruct_adj(neighs, weight):
    from scipy.sparse import csr_matrix
    import numpy as np
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    w = weight[:,1:].reshape((row.shape[0],))
    w = 1/w
    adj = csr_matrix((w, (row, col)), shape=(dim, dim))
    return adj
