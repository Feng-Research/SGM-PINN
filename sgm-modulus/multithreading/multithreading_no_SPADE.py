import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import pdb
import numpy as np
import os 
import time
import modulus.external.coarsen_utils as GraphUtils
from scipy.sparse import csc_matrix
from modulus.external.Clustering.C_HyperEF import C_HyperEF
import gc


##This module is an older version that will be fully removed when julia code is replaced with Python code

def worker_HyperEF_fn(wID,
                        hyper_queue_cmd, hyper_queue, 
                        knn, sparsification,
                        level, r, initial_graph_vars,
                        graph_vars):
        JLoad = C_HyperEF()
        start = time.time()
        count = 0 ##TODO Change this to keep track of # of runs
        print("Opened",flush=True)
        while True:
            try:
                message = hyper_queue_cmd.get(timeout=2)
                print(f"{wID} Message get",flush=True)
                count = 0
            except Empty:
                if count == 15:
                    print(f'{wID} EMPTYWORKEREMPTY', flush = True)
                    count = 0
                count += 1
                time.sleep(30) 
                continue
            except:
                print('Queue destroyed?')
                exit()
            if message[1] == 'STOP':
                print("Break",flush=True)
                break
            assert(wID == message[0][0])
            print(f"{wID} Creating Subgraph",flush=True)
            #Create new sub graph
            dataset = message[1]
            print(f"Message get {type(dataset)} {type(dataset['x'])}",flush=True)
            if message[0][1] == 'initial':
                construct_vars = initial_graph_vars
            else:
                construct_vars = graph_vars
            print(f"Message get {message[0][1]}")
            clusters = JLoad.HyperEF_P(dataset,
                construct_vars,
                knn,
                level)
            print("Pushing to Queue",flush = True)
            while True:
                try:
                    hyper_queue.put_nowait(np.array([wID,clusters], dtype="object"))
                    break
                except:
                    print("Put Failed, retrying", flush = True)
            print("Push Success", flush = True)
            end = time.time()
            del clusters
            del dataset
            gc.collect()
            print(f'{wID} TIME TAKEN {end-start}', flush = True)
            break #### julia in multiprocessing has issues releasing resources, need to kill/restart the workers.


import multiprocessing as mp
from multiprocessing.queues import Empty
class MultiProcessHyperEF:

    def __init__(self, dataset, batch_size, 
                 knn, sparsification,
                 level, r, initial_graph_vars,
                 graph_vars, grid_width = 2):
        self.dataset = dataset ##TODO pass the fcn to get invars instead?
        self.batch_size = batch_size
        self.knn = knn
        self.sparsification = sparsification
        self.level = level
        self.r = r
        self.initial_graph_vars = initial_graph_vars
        self.graph_vars = graph_vars
        self.grid_width = grid_width

        self.workers_hyper = [] #workers for graph construction
        self.workers_importance = [] #workers for importance sampling
        self.binned_datasets_refs = MultiProcessHyperEF.split_grids(self.dataset, self.grid_width) # this would be a list of lists of indexes for samples in each grid space
        
        for k,i in enumerate(self.binned_datasets_refs):
            np.savez(f'{k}' +'_initialRefs.npz', i)

        self.num_workers = len(self.binned_datasets_refs)
        self.binned_datasets = []
        self.binned_clusters = [None for _ in range(self.num_workers)]
        self.combined_graph = None
 
        self.hyper_queues_cmd = []
        self.importance_queues_cmd = []
        self.hyper_queue_out = mp.Queue()
        self.importance_queue_out = mp.Queue()

        self.new_graph_progress = np.zeros(self.num_workers)
        self.grid_importance = [] # keep track of total grid importance, reduce how often it's refreshed relative to others
        self.running = False



        for wID in range(self.num_workers):
            hyper_queue_cmd = mp.Queue()
            worker = mp.Process(
                target=worker_HyperEF_fn, args=(wID, 
                                                hyper_queue_cmd, self.hyper_queue_out, 
                                                self.knn, self.sparsification, self.level,
                                                self.r, self.initial_graph_vars, 
                                                self.graph_vars)
            )
            worker.daemon = True
            worker.start()
            self.workers_hyper.append(worker)
            self.hyper_queues_cmd.append(hyper_queue_cmd)

    def update_dataset(self, dataset):
        self.dataset = dataset
        np.savez(f'./wholeData_{time.time()}',self.dataset)
        print('UPDATED DATASET')

    def new_graph(self, detail = None, manual = None):
        self.binned_datasets = [{k:v[subset] for k,v in self.dataset.items()} for subset in self.binned_datasets_refs]
        print('RE-BINNED DATASETS')
        count = 0
        for wID in range(self.num_workers):
            worker = mp.Process(
                target=worker_HyperEF_fn, args=(wID, 
                                                self.hyper_queues_cmd[wID], self.hyper_queue_out, 
                                                self.knn, self.sparsification, self.level,
                                                self.r, self.initial_graph_vars, 
                                                self.graph_vars)
            )
            worker.daemon = True
            worker.start()
            self.workers_hyper[wID] = (worker)

        if manual == None:
            for i in range(self.num_workers): #TODO if threshold for importance is met
                count += 1
                print(f'Putting {i},{detail}')
                self.hyper_queues_cmd[i].put(np.array([(i,detail),self.binned_datasets[i]], dtype = "object"), 
                        timeout = 5)
        else:
            for i in manual: #TODO if threshold for importance is met
                count += 1
                print(f'Putting {i},{detail}')
                self.hyper_queues_cmd[i].put(np.array([(i,detail),self.binned_datasets[i]], dtype = "object"), 
                        timeout = 5)
        print("Launched HyperEF")
        self.running = True
        return count
    
    def clear_queues(self):
        print('clearing queues')
        for i in range(self.num_workers): 
            print(f'Clearing {i}')
            self.hyper_queues_cmd[i].put(np.array([i,'clear'], dtype = "object"))
    
    def next_laplacian(self):
        print('checking queues')
        print(f'graph_progress: {self.new_graph_progress}')
        for i in range(self.num_workers): #limit no. of checks (don't while loop) so training can continue while waiting
            try:
                message = self.hyper_queue_out.get(timeout=1)
                wID = message[0]
                print(f"Recieved Clusters for {wID}")
                print(type(message[1]))
            except Empty:
                print(f'Attempt {i} NoMsg, ', flush = True)
                break
            if type(message[1]).__name__ == 'csc_matrix': #for eventual thresholding TODO
                self.binned_clusters[wID] = message[1]
            elif message[1] == 'skip':
                print(f'Skip {wID}')
            self.new_graph_progress[wID] = 1
        print(f'graph_progress: {self.new_graph_progress}')
        if not all(self.new_graph_progress == 1):
            return False ##caller will get rand samples instead
        print('COMBINE/RESET')
        self.new_graph_progress = np.zeros(self.num_workers) #reset new graph progress
        self.running = False
        self.combined_graph = self.combine_grids_L(self.binned_clusters, self.binned_datasets_refs) #combine graphs
        np.savez(f'./combinedGraph_{time.time()}',np.array(['Interior',self.combined_graph],dtype='object'))
        return True
    
    @staticmethod
    def combine_grids_L(binned_clusters,binned_refs): #combine the direct L outputs of new_graph
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