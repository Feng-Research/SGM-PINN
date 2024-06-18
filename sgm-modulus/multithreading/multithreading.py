import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import numpy as np
import os 
import time
from scipy.sparse import csc_matrix
import modulus.external.SPADE_score.SPADE_nxFree as SPADE
from modulus.external.Clustering.C_HyperEF import C_HyperEF
import modulus.external.SGM_SPADE_utils as GraphUtils
import gc
from sklearn import preprocessing
import atexit

#This module implements a multiprocessing manager
#Julia called from python running in a subprocess may throw errors when interrupted -- 
#still an open issue on the julia git -- additionally subprocesses demonstrated memory bloat over time
#To be replaced with full python implementations.

#convert dictionary to numpy with relevant info
def dict_to_numpy(invar):
        data = np.concatenate([invar[x] for x in invar.keys()], axis=1)
        num_elements = list(invar.values())[0].shape[0]
        ids = np.arange(num_elements)
        lkeys = invar.keys()
        dim = len(lkeys)
        return data, ids, lkeys, num_elements, dim

#scale spade scores
def SPADE_uniform_quantile(scores):
        return preprocessing.normalize(
                                [preprocessing.quantile_transform(
                                    scores.reshape(-1, 1), 
                                    n_quantiles=300, 
                                    output_distribution='uniform', 
                                    ignore_implicit_zeros=False, 
                                    subsample=100000, 
                                    random_state=None).reshape(-1,)],
                                norm='l2', 
                                axis=1, copy=True, return_norm=False
                            ).reshape(-1,)**2

#Try to check if main still exists before trying to push. (Not a full fix.)
def send_data_protected(wID,pwID,return_pipe,data):
    print(f"{pwID} Pushing to main Pipe",flush = True)
    if return_pipe.poll(timeout = .1):
        print(f"{pwID} Pipe Gone, exiting",flush = True)
        sys.exit() #Main doesn't send data here, pipe reports a value ready if the other end dies. If value ready on this end, main died.
    return_pipe.send(np.array([wID,data], dtype="object")) ##NOTE blocking call
    print(f"{pwID} Push Success", flush = True)

#sparsifier subprocess
def worker_HyperEF_fn(wID, detail,                      
                        dataset, 
                        return_pipe, 
                        knn, sparsification,
                        level, r, 
                        initial_graph_vars,
                        graph_vars
                    ):
        pwID = f"CLUSTER_{wID}"
        JLoad = C_HyperEF()
        start = time.time()
        print(f"{pwID} Opened",flush=True)
        print(f"{pwID} Creating Subgraph",flush=True)
        print(f"{pwID} Dataset get  {type(dataset)} {type(dataset['x'])}",flush=True)
        if detail == 'initial':
            construct_vars = initial_graph_vars
            print(f"{pwID} Detail message get {detail}")
        else:
            construct_vars = graph_vars
        clusters = JLoad.HyperEF_P(dataset,
            construct_vars,
            knn,
            level)
        print(f'{pwID} CLUSTERS::: {clusters.shape}', flush = True)
        send_data_protected(wID,pwID,return_pipe,clusters)
        end = time.time()
        print(f'{pwID} TIME TAKEN {end-start}', flush = True)

#SPADE, stability score subprocess
def worker_SPADE_fn(wID,
                    detail,
                    dataset_input,
                    dataset_output,
                    return_pipe, 
                    knn, num_eigs,
                    initial_graph_vars,
                    output_graph_vars,
                    ):
        start = time.time()
        pwID = f"SPADE_{wID}"
        print(f"{pwID} Opened",flush=True)
        print(f"{pwID} Creating Subgraph",flush=True)
        #Create new sub graph
        print(f"{pwID} Dataset input get  {type(dataset_input)} {type(dataset_input[list(dataset_input.keys())[0]])}",flush=True)
        print(f"{pwID} Dataset output get  {type(dataset_output)} {type(dataset_output[list(dataset_output.keys())[0]])}",flush=True)
        print(f"{pwID} Dataset output keys: {list(dataset_output.keys())}")
        print(f"{pwID} Dataset initial_graph_vars:{initial_graph_vars}")
        print(f"{pwID} Dataset output_graph_vars:{output_graph_vars}")
        if detail == 'initial':
            dataset_input = {k:v.cpu().detach().numpy() for k,v in dataset_input.items() if k in initial_graph_vars}
            print(f"{pwID} Dataset Input Detached, Initial Graph Vars",flush=True)
        else:
            dataset_input = {k:v for k,v in dataset_input.items() if k in output_graph_vars}
            print(f"{pwID} Dataset Input As Received, main Graph Vars",flush=True)
        start2 = time.time()
        dataset_input, _, lkeys, num_elements, dim = dict_to_numpy(dataset_input)
        end2 = time.time()
        print(f'{pwID} DICT_TO_NUMPY_INPUT {end2-start2}', flush = True)
        print(f'{pwID}Keys:' + str(lkeys))
        print(f'Size:({num_elements}x{dim})')
        dataset_output = {k:v for k,v in dataset_output.items() if k in output_graph_vars}
        dataset_output, _, lkeys, num_elements, dim = dict_to_numpy(dataset_output)
        print(f'{pwID}Using: {str(lkeys)}, KNN: {knn}, num_eigs: {num_eigs}')
        print(f'{pwID}Keys:' + str(lkeys))
        print(f'Size:({num_elements}x{dim})')

        dataset_input = preprocessing.scale(dataset_input)
        dataset_output = preprocessing.scale(dataset_output)
        
        spadeOut = SPADE.spade(dataset_input, dataset_output, 
                            k=knn, 
                            num_eigs = num_eigs,
                            sparse = False, weighted=False,
                            wID = wID)[:4]
        
        spadeOut = np.array(spadeOut, dtype="object")
        
        outs = ['TopEig', 'TopEdgeList', 'TopNodeList', 'node_score', 'L_in', 'L_out', 'Dxy', 'Uxy']
        
        send_data_protected(wID,pwID,return_pipe,spadeOut)
        end = time.time()
        print(f'{pwID} TIME TAKEN {end-start}', flush = True)

#batch subprocess 
def worker_BATCH_fn(wID,
                    CLUSTER_combined_results, 
                    cluster_subsets,
                    subset_values,
                    return_pipe,
                    recieve_pipe,
                    sMin, sMax,
                    total,
                    shuffle,
                    avg,
                    lkeys
                    ):
        while True:
            start = time.time()
            pwID = f"BATCH_{wID}"
            print(f"{pwID} Opened",flush=True)
            print(f"{pwID} Creating Subgraph",flush=True)
            print(f"{pwID} P get  {type(CLUSTER_combined_results)} {CLUSTER_combined_results.shape}",flush=True)
            print(f"{pwID} Subsets get  {type(cluster_subsets)} {cluster_subsets.shape}",flush=True)
            print(f"{pwID} Values get  {type(subset_values)} {subset_values.shape}",flush=True)
            print(f'{pwID} Using: {str(lkeys)}, min/max: {(sMin,sMax)}, total: {total}, shuffle: {shuffle}, avg:{avg}')
            batches = GraphUtils.getMappingCore_weighted_fast_modulus(CLUSTER_combined_results, 
                                                        cluster_subsets,
                                                        subset_values,
                                                        sMin = sMin, sMax = sMax,
                                                        total = total, 
                                                        shuffle = True,
                                                        avg = True)
            
            send_data_protected(wID,pwID,return_pipe,batches)
            end = time.time()
            print(f'{pwID} BATCH TIME TAKEN {end-start}', flush = True)
            if recieve_pipe.poll(timeout = .1):
                break

import multiprocessing as mp
class MultiProcessSPADE:

    def __init__(self, dataset, batch_size, 
                 knn, initial_graph_vars, graph_vars, 
                 SPADE_vars,LOSS_vars,
                 grid_width = 1,
                 num_eigs = 2,
                 SPADE_GRID = False,
                 level = 4,
                 sparsification = 0,
                 r = 0,
                 sample_ratio = .10,
                 sample_bounds = [5/100,70/100]
                 ):
        self.dataset = dataset
        self.dataset_output = None
        self.batch_size = batch_size
        self.knn = knn
        self.num_eigs = num_eigs
        self.SPADE_GRID = SPADE_GRID    

        self.level = level
        self.sparsification = sparsification
        self.r = r
        self.sample_ratio=sample_ratio
        self.sample_bounds=sample_bounds

        self.initial_graph_vars = initial_graph_vars
        self.graph_vars = graph_vars
        self.SPADE_vars = SPADE_vars
        print(f'SPADE_vars: {self.SPADE_vars}')
        self.LOSS_vars = LOSS_vars
        print(f'LOSS_vars: {self.LOSS_vars}')
        self.grid_width = grid_width

        
        self.binned_datasets_refs = self.split_grids(self.dataset, self.grid_width) # this would be a list of lists of indexes for samples in each grid space
        
        for k,i in enumerate(self.binned_datasets_refs):
            np.savez(f'{k}' +'_initialRefs.npz', i)

        self.num_workers = len(self.binned_datasets_refs)
        self.SPADE_num_workers = self.num_workers**SPADE_GRID

        self.CLUSTER_workers = [None for _ in range(self.num_workers)] #workers for sparsification
        self.SPADE_workers = [None for _ in range(self.SPADE_num_workers)] #workers for SPADE calcs
        self.BATCH_workers = [None for _ in range(1)] #workers for creating sampling batches

        self.binned_datasets_in = []
        self.binned_datasets_out = []

        self.SPADE_binned_results = [None for _ in range(self.SPADE_num_workers)]
        self.SPADE_combined_results = None

        self.CLUSTER_binned_results = [None for _ in range(self.num_workers)]
        self.CLUSTER_combined_results = None
 
        self.CLUSTER_pipes_recv = [None for _ in range(self.num_workers)]
        self.SPADE_pipes_recv = [None for _ in range(self.SPADE_num_workers)]
        self.BATCH_pipes_recv = [None for _ in range(2)]

        self.CLUSTER_progress = np.zeros(self.num_workers)
        self.SPADE_progress = np.zeros(self.SPADE_num_workers)

        self.grid_importance = [] ##TODO keep track of total grid importance, reduce how often it's refreshed relative to others?
        self.running = 0

        self.cluster_subsets = None
        self.subset_values = None

        self.hitCount=1
        self.missCount=1


        #Kill all subprocesses the manager knows of on exit (may not catch crashes or errors)
        def exit_handler():
            print('EXIT HANDLE')
            counts = 0
            countf = 0
            for i in [self.CLUSTER_workers + self.SPADE_workers + self.BATCH_workers]:
                try:
                    i.kill()
                    counts += 1
                except:
                    countf += 1
            print(f'Stopped:{counts}, N/a {countf}')

        atexit.register(exit_handler)


    #launch SPADE process
    def start_SPADE_Workers(self, detail, dataset_input, dataset_output):
        for wID in range(self.SPADE_num_workers):
            pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
            worker = mp.Process(
                target=worker_SPADE_fn, args=(wID, detail,
                                            dataset_input[wID], dataset_output[wID],
                                            pipe_fromThread_send, 
                                            self.knn, self.num_eigs,
                                            self.initial_graph_vars, 
                                            self.SPADE_vars)
            )
            worker.daemon = True
            worker.start()
            self.SPADE_workers[wID] = worker
            self.SPADE_pipes_recv[wID] = pipe_fromThread_recv

    #launch sparsifier processes
    def start_CLUSTER_Workers(self, detail, dataset):
        for wID in range(self.num_workers):
            pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
            worker = mp.Process(
                target=worker_HyperEF_fn, args=(wID, detail,
                                            dataset[wID],
                                            pipe_fromThread_send, 
                                            self.knn, self.sparsification,
                                            self.level, self.r, 
                                            self.initial_graph_vars,
                                            self.graph_vars)
            )
            worker.daemon = True
            worker.start()
            self.CLUSTER_workers[wID] = worker
            self.CLUSTER_pipes_recv[wID] = pipe_fromThread_recv
    
    #launch batch process
    def start_BATCH_worker(self, wID, 
                            CLUSTER_combined_results, 
                            cluster_subsets,
                            subset_values,
                            sMin, 
                            sMax,
                            total,
                            shuffle,
                            avg,
                            lkeys):
        if self.BATCH_workers[0]:
            print(f'BATCH WORKERS RESETTING: {self.BATCH_workers[0]}')
            self.BATCH_workers[0].kill() #there won't be another attempt to read until this fcn is done and overwrites the pipes, 
                                           #this should be OK
        pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
        pipe_fromMain_recv, pipe_fromMain_send = mp.Pipe()
        worker = mp.Process(
            target=worker_BATCH_fn, args=(wID,
                                        CLUSTER_combined_results, 
                                        cluster_subsets,
                                        subset_values,
                                        pipe_fromThread_send, 
                                        pipe_fromMain_recv,
                                        sMin, sMax,
                                        total,
                                        shuffle,
                                        avg,
                                        lkeys)
        )
        worker.daemon = True
        worker.start()
        self.BATCH_workers[0] = worker
        print(f'BATCH WORKERS SET: {self.BATCH_workers[0]}')
        self.BATCH_pipes_recv[0] = pipe_fromThread_recv
        self.BATCH_pipes_recv[1] = pipe_fromMain_send


    def update_dataset(self, dataset, counter):
        self.dataset = dataset
        np.savez(f'./wholeData_in{counter}',self.dataset)
        print('UPDATED DATASET')
    
    def update_dataset_outputs(self, dataset, counter):
        if self.dataset_output:
            print('PUSH-BACK LAST DATASET_OUTPUT')
            self.dataset = self.dataset_output
        self.dataset_output = dataset
        np.savez(f'./wholeData_out{counter}',dataset)
        print('UPDATED DATASET_OUTPUT')
    
    def new_clusters(self, detail = None):
        self.binned_datasets_out = [{k:v[subset] for k,v in self.dataset_output.items()} for subset in self.binned_datasets_refs]
        print('RE-BINNED DATASETS')
        print(f'Starting {self.num_workers} CLUSTER Workers, Details: {detail}')
        self.start_CLUSTER_Workers(detail, self.binned_datasets_out)
        print("Launched HyperEF")
        self.running = 1
    
    def new_spade_scores(self, detail = None):
        if self.SPADE_GRID:
            self.binned_datasets_in = [{k:v[subset] for k,v in self.dataset.items() if k in self.SPADE_vars} 
                                       for subset in self.binned_datasets_refs ]
            self.binned_datasets_out = [{k:v[subset] for k,v in self.dataset_output.items() if k in self.SPADE_vars} 
                                        for subset in self.binned_datasets_refs]
            raise NotImplementedError
        else:
            if [i for i in self.dataset.keys() if i in self.SPADE_vars]:
                dataset = {k:v[self.cluster_subsets[:,0]] for k,v in self.dataset.items() if k in self.SPADE_vars}
            else:
                print('No SPADE_vars in dataset')
                dataset = {k:v[self.cluster_subsets[:,0]] for k,v in self.dataset.items()}
            dataset_output = {k:v[self.cluster_subsets[:,0]] for k,v in self.dataset_output.items() if k in self.SPADE_vars}
        print(f'Starting {self.SPADE_num_workers} SPADE Workers, Details: {detail}')
        self.start_SPADE_Workers(detail, [dataset], [dataset_output])
        print("Launched SPADE")
        self.running = 2

    ##Mappings: subsets to their clusters, clusters to the whole dataset.
    ##cluster_subsets, sample_ratio % of each cluster. subset_samples, the corresponding residuals. SPADE is always calculated on the subsets.
    ##new_clusters use the entire dataset. mini-scores should update residuals only for the relevant subset.

    #Refreshes importance scores of each cluster with a subset from each. Starts new batch worker. Uses last major set of SPADE scores.
    def subset_refresh(self, update_fcn, total,
                        shuffle = True, avg = True):
        print('REFRESHING SUBSET VALUES')
        print(f'Total: {total}')
        
        refresh_importance_collect, refresh_importance_vars = update_fcn(self.prev_cluster_subsets)
        cols = [k for k,v in enumerate(refresh_importance_vars) if v in self.LOSS_vars]
        if cols:
            self.prev_subset_values = np.concatenate((refresh_importance_collect[:,cols], #for now just updating raw scores
                                        self.SPADE_combined_results.reshape(-1,1)), 
                                        axis=1)
        else:
            self.prev_subset_values = self.SPADE_combined_results.reshape(-1,1)
            
        self.start_BATCH_worker(0,
                            self.prev_CLUSTER_combined_results, 
                            self.prev_cluster_subsets,
                            self.prev_subset_values,
                            self.sample_bounds[0], self.sample_bounds[1], 
                            total, shuffle, avg, self.LOSS_vars)
        

    #Fully refreshes SPADE and Importance scores.
    def subset_data_collect(self, importance_collect, importance_vars, 
                            total,
                            shuffle = True, avg = True):
        cols = [k for k,v in enumerate(importance_vars) if v in self.LOSS_vars]
        if cols:
            self.subset_values = np.concatenate((importance_collect[self.cluster_subsets[:,0]][:,cols], 
                                        self.SPADE_combined_results.reshape(-1,1)), 
                                        axis=1)
            assert all([np.all(importance_collect[self.cluster_subsets[:,0]][:,importance_vars.index(i)] 
                           == self.subset_values[:,k]) for k,i in enumerate(self.LOSS_vars)])
        else:
            self.subset_values = self.SPADE_combined_results.reshape(-1,1)
        try:
            print(f'New subset vals: {self.prev_subset_values.shape},{self.subset_values.shape}')
            print(f'New cluster subsets:{self.prev_cluster_subsets.shape},{self.cluster_subsets.shape}')
            print(f'New combined subsets:{self.prev_CLUSTER_combined_results.shape},{self.CLUSTER_combined_results.shape}')
        except:
            print('First Pass')
        print(f'Batch Collect Total: {total}')
        self.start_BATCH_worker(0,
                            self.CLUSTER_combined_results, 
                            self.cluster_subsets,
                            self.subset_values,
                            self.sample_bounds[0], self.sample_bounds[1], 
                            total, shuffle, avg, self.LOSS_vars)
        
        self.prev_subset_values = self.subset_values
        self.prev_cluster_subsets = self.cluster_subsets
        self.prev_CLUSTER_combined_results = self.CLUSTER_combined_results


    #Use random sampling when batches aren't ready.
    def batch_receiver(self, batch_iterations):
        print(f'Hits:{self.hitCount}, Misses:{self.missCount}, Ratio: {self.hitCount/(self.missCount+self.hitCount)}')
        print('CHECKING FOR BATCH')
        print(f'{self.BATCH_pipes_recv[0]}')
        if self.BATCH_pipes_recv[0].poll(timeout = .3):
            self.hitCount+=batch_iterations
            message = self.BATCH_pipes_recv[0].recv()
            wID = message[0]
            print(f"Recieved Batch from {wID}")
            print(type(message[1]))
            print(f'pipe: {self.BATCH_pipes_recv}')
            return message[1]
        else:
            self.missCount+=100
            print(f'Batch reciever no data', flush = True)
            current_dataset_size = list(self.dataset.values())[0].shape[0]
            return np.random.choice(range(current_dataset_size),size=int(self.batch_size*100),replace=False)
    
    #Checks for status of sparsification and SPADE scoring (large T_g rebuilds)
    def check_graph_sequential(self, counter, detail = None):
        if not self.running:
            print(f'Graph not running, skipping check.')
            return False
        
        #stage 1, clustering

        if self.running == 1:
            print('checking CLUSTERING queues')
            print(f'CLUSTER_progress: {self.CLUSTER_progress}, Detail:{detail}')
            if not (all(self.CLUSTER_progress == 1)):
                for i in range(self.num_workers): #limit no. of checks (don't while loop) so training can continue while waiting
                    
                    if self.CLUSTER_progress[i] == 1:
                        print(f'Cluster {i} Recieved, waiting')
                        continue ##check next
                    if self.CLUSTER_pipes_recv[i].poll(timeout = .1):
                        message = self.CLUSTER_pipes_recv[i].recv()
                        wID = i
                        print(f"Recieved Clusters for {wID}")
                        print(type(message[1]))
                        self.CLUSTER_binned_results[wID] = message[1]
                        self.CLUSTER_progress[wID] = 1
                    else:
                        print(f'Cluster attempt {i} NoMsg, ', flush = True)
                        continue ##no message, check next
            print(f'CLUSTER_progress: {self.CLUSTER_progress}')
            if not (all(self.CLUSTER_progress == 1)):
                return False
            print(f'CLUSTER_progress: {self.CLUSTER_progress}')
            self.CLUSTER_combined_results = self.combine_grids_L(self.CLUSTER_binned_results, self.binned_datasets_refs) #combine cluster grids (Laplacian matrix)
            ##sub-cluster
            self.cluster_subsets = GraphUtils.getMappingCore(self.CLUSTER_combined_results, total = 1, 
                                                                    repetition = True, shuffle = False, sample = self.sample_ratio)
            self.new_spade_scores(detail=detail)

        #stage 2, SPADE

        print(f'SPADE_progress: {self.SPADE_progress}')
        if not (all(self.SPADE_progress == 1)):
            for i in range(self.SPADE_num_workers): #limit no. of checks (don't while loop) so training can continue while waiting
                if self.SPADE_progress[i] == 1:
                    print(f'SPADE grid {i} Recieved, waiting') #cont to CLUSTER
                else:
                    if self.SPADE_pipes_recv[i].poll(timeout = .1):
                        wID = i
                        print(f"Recieved SPADE for {wID}")
                        message = self.SPADE_pipes_recv[i].recv()
                        print(type(message[1]))
                        ##TopEig = message[1][0]
                        ##TopEdgeList = message[1][1]
                        ##TopNodeList = message[1][2]
                        node_score = message[1][3]
                        self.SPADE_binned_results[wID] = node_score
                        self.SPADE_progress[wID] = 1
                    else:
                        print(f'SPADE attempt {i} NoMsg, ', flush = True)
        print(f'SPADE_progress: {self.SPADE_progress}')
        if not (all(self.SPADE_progress == 1)):
            return False 
        print('SPADE COMPLETE/RESET')
        self.CLUSTER_progress = np.zeros(self.num_workers) #RESET
        self.SPADE_progress = np.zeros(self.SPADE_num_workers) #RESET
        self.running = False
        self.SPADE_combined_results = self.combine_grids_nodes(self.SPADE_binned_results, self.binned_datasets_refs) #combine SPADE grids
        if False: #for debug
            np.savez(f'./combinedGrids_{counter}',{'SPADE':self.SPADE_combined_results,
                                                'CLUSTER':self.CLUSTER_combined_results,
                                                'dataset_in':self.dataset,
                                                'dataset_out':self.dataset_output})
        with open('./logging.txt', 'a+') as f:
                counts = np.array([self.CLUSTER_combined_results.indptr[i+1]-self.CLUSTER_combined_results.indptr[i]
                          for i in range(0,self.CLUSTER_combined_results.shape[1])])
                f.write(f'P shape:{self.CLUSTER_combined_results.shape},SPADE shape:{self.SPADE_combined_results.shape} \n')
                f.write(f'Min: {np.min(counts)}, Max {np.max(counts)}, Mean {np.mean(counts)}, Median {np.median(counts)} \n')
                bins = range(np.min(counts),np.max(counts),max(1,int(np.max(counts)/10)))
                histogram = np.histogram(counts,bins)
                f.write(f'Histogram: {histogram[0]},{histogram[1]}\n')
        if False: #for debug
            for i in range(len(self.binned_datasets_refs)):
                np.savez(f'./Step{counter}_GridRefs_{i}',np.array(['Interior',self.binned_datasets_refs[i]],dtype='object'))
                np.savez(f'./Step{counter}_GridIn_{i}',np.array(['Interior',self.binned_datasets_in[i]],dtype='object'))
                np.savez(f'./Step{counter}_GridOut_{i}',np.array(['Interior',self.binned_datasets_out[i]],dtype='object'))
                np.savez(f'./Step{counter}_GridSPADE_{i}',np.array(['Interior',self.SPADE_binned_results[i]],dtype='object'))
        return True

    @staticmethod
    def combine_grids_nodes(binned_node_scores, binned_refs): #combine the spade scores of all nodes in all grids in original order.
        print(f'COMBINEDGRIDSNODES{len(binned_node_scores)}')
        if len(binned_node_scores) == 1:
            return SPADE_uniform_quantile(binned_node_scores[0])
        grid_node_scores = np.concatenate(binned_node_scores, axis=0) #all scores
        refs_all_grids = np.concatenate(binned_refs, axis=0) #a list of n indexes, in the same order as the scores, with the index for that score in the original dataset
        return SPADE_uniform_quantile(grid_node_scores[np.argsort(refs_all_grids)])
    
    #recombines grid's cluster arrays back into a single matrix
    @staticmethod
    def combine_grids_L(binned_clusters,binned_refs): #combine the direct L outputs of new_graph
        binned_clusters = [i.tocsc() for i in binned_clusters]
        if len(binned_clusters) ==  1:
            return binned_clusters[0].tocsc()
        rows = 0
        cols = 0
        for i in binned_clusters:
            rows += i.shape[0]
            cols += i.shape[1]
        indptr = []
        indices = []
        data = []
        for i in range(len(binned_refs)):      
            indptr += [(binned_clusters[i].indptr + int(np.sum([len(prev) for prev in indices])))[int(i>0)::]]
            indices += [binned_refs[i][binned_clusters[i].indices]]
            data += [binned_clusters[i].data]
        cc = lambda x: np.concatenate(x,axis=0)
        return csc_matrix((cc(data), cc(indices), cc(indptr)), shape=(rows,cols))
    
    #splits input data into grids
    @staticmethod
    def split_grids(dataset, grid_width):
        values = {}
        ranges = {}
        dkeys = [i for i in ['x','y','z'] if i in dataset.keys()]
        if dkeys == []:
            raise ValueError('No x,y,z data')
        for cord in dkeys:
            values[cord] = dataset[cord].cpu().detach().numpy()
            ranges[cord] = (np.min(dataset[cord].cpu().detach().numpy()),
                            np.max(dataset[cord].cpu().detach().numpy()),
                            np.ptp(dataset[cord].cpu().detach().numpy()))
        for k,cord in enumerate(dkeys):
            values[cord] = values[cord] - ranges[cord][0] # shift to start at 0
            if k == 1:
                values[cord] = np.clip(grid_width - 1 - (values[cord]/((ranges[cord][2]/grid_width))).astype(int), 0, grid_width-1) # grid coord of each point
            else:
                values[cord] = np.clip((values[cord]/((ranges[cord][2]/grid_width))).astype(int), 0, grid_width-1) # grid coord of each point
        grids = np.zeros((values['x'].shape[0]),dtype=np.uint32) #there should always be at least x
        #x+grid_width*y+grid_width^2*z is the box number
        for k, cord in enumerate(dkeys):
            grids = np.add(grids,np.multiply(values[cord].reshape(values[cord].shape[0]),int(grid_width**k)))
        blocks = [[] for _ in range(0,grid_width**len(dkeys))]
        for d,b in enumerate(grids):
            blocks[b].append(d)
        return [np.array(i) for i in blocks]
