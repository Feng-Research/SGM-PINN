# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Modulus Dataset constructors for continuous type data
"""

from typing import Dict, List, Callable, Union, Tuple

import numpy as np
import time

from modulus.utils.io.vtk import var_to_polyvtk
from .dataset import Dataset, IterableDataset, _DictDatasetMixin


class _DictPointwiseDatasetMixin(_DictDatasetMixin):
    "Special mixin class for dealing with dictionaries as input"

    def save_dataset(self, filename):

        named_lambda_weighting = {
            "lambda_" + key: value for key, value in self.lambda_weighting.items()
        }
        save_var = {**self.invar, **self.outvar, **named_lambda_weighting}
        var_to_polyvtk(filename, save_var)

    

class DictPointwiseDataset(_DictPointwiseDatasetMixin, Dataset):
    """A map-style dataset for a finite set of pointwise training examples."""

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        lambda_weighting: Dict[str, np.array] = None,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length


class DictInferencePointwiseDataset(Dataset):
    """
    A map-style dataset for inferencing the model, only contains inputs
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        output_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.output_names = output_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return (invar,)

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.output_names)


class ContinuousPointwiseIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of pointwise training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        lambda_weighting_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }

        def iterable_function():
            while True:
                invar = Dataset._to_tensor_dict(self.invar_fn())
                outvar = Dataset._to_tensor_dict(self.outvar_fn(invar))
                lambda_weighting = Dataset._to_tensor_dict(
                    self.lambda_weighting_fn(invar, outvar)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        invar = self.invar_fn()
        return list(invar.keys())

    @property
    def outvar_keys(self):
        invar = self.invar_fn()
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):
    """
    An infinitely iterable dataset that applies importance sampling for faster more accurate monte carlo integration
    """

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure

        def iterable_function():

            # TODO: re-write idx calculation using pytorch sampling - to improve performance

            counter = 0
            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    print('RESAMPLE')
                    print(counter)
                    print(self.resample_freq)
                    list_importance = []
                    list_invar = { ### variable: list of batches 
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in self.invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))): ###getting number of elements in a batch, should all be the same
                        importance = self.importance_measure(  ##per batch importance measure, returns an array the same size as the batch
                            {key: value[i] for key, value in list_invar.items()} ###variable: batch i
                        )
                        list_importance.append(importance) ###list of arrays of importances of each batch
                    importance = np.concatenate(list_importance, axis=0) ###reform list of lists into a Bsize*Bnum by 1 vector, previous batching was to keep calc on GPU?
                    prob = importance / np.sum(self.invar["area"].numpy() * importance) ###np array of probabilities for each batch? importance/(area*importance of each sample, summed)
                    #probability is importance of a sample divided by area-weighted importance of the batch

                # sample points from probability distribution and store idx
                idx = np.array([])
                while True:
                    r = np.random.uniform(0, np.max(prob), size=self.batch_size) ### batch_size array of random vars from 0 to max(prob)
                    try_idx = np.random.choice(self.length, self.batch_size)     ### batch_size array of indexes from dataset
                    if_sample = np.less(r, prob[try_idx, :][:, 0])               ### true/false array of indexes where r < prob
                    idx = np.concatenate([idx, try_idx[if_sample]])              ### add true indexes to the sampling indexes
                    if idx.shape[0] >= batch_size:
                        idx = idx[:batch_size]
                        break
                idx = idx.astype(np.int64)                                       

                # gather invar, outvar, and lambda weighting
                invar = _DictDatasetMixin._idx_var(self.invar, idx)
                outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
                lambda_weighting = _DictDatasetMixin._idx_var(
                    self.lambda_weighting, idx
                )

                # set area value from importance sampling
                invar["area"] = 1.0 / (prob[idx] * batch_size)

                # return and count up
                counter += 1
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

class DictGraphImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):
    """
    Graph-Based Importance Sampling
    An infinitely iterable dataset that implements SGM-PINN with SPADE scoring, used for the AR example.
    """

    def warmup_loop(self, indicator_fcn=None,threshold=None,
                    detail = None,
                    message = 'WARMUP SAMPLING'): ##indicator must take self.counter as an input
        if indicator_fcn == None and threshold == None:
            indicator_fcn = lambda x, detail = None: x
            threshold = self.warmup
        indicator = indicator_fcn(self.counter, detail = detail)
        while indicator < threshold: ##warmup before trying to get outputs from NN
            warmup_idx = np.split(np.random.permutation(range(0,self.length)), self.length // self.batch_size)
            print(f'Step: {self.counter}: {message}, Detail:{detail}')
            for i in warmup_idx:
                invar = _DictDatasetMixin._idx_var(self.invar, i)
                outvar = _DictDatasetMixin._idx_var(self.outvar, i)
                lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, i)
                self.counter += 1
                if self.counter%1000 == 0:
                    print("Checking Indicator")
                    indicator = indicator_fcn(self.counter, detail=detail)
                    print(f'Indicator: {indicator}, {threshold}, {indicator<threshold}')
                    if not (indicator < threshold):
                        break
                yield (invar, outvar, lambda_weighting)
            

    def running_loop(self):
        pass
    
    #snapshot of all outputs from importance functions
    def importance_update(self):
        ##Get output data as np.
        print(f'Calculating current outputs for clustering...{self.graph_vars} out of {self.invar.keys()}')
        list_invar = {
            key: np.split(value, value.shape[0] // self.batch_size)
            for key, value in _DictDatasetMixin._idx_var(self.invar, range(0,self.length)).items()
        }
        importance_collect = []
        importance_vars = []
        for i in range(len(next(iter(list_invar.values())))):
            importance = self.importance_measure(
                {key: value[i] for key, value in list_invar.items()}
            )
            importance_collect.append(importance[0])
            if not importance_vars:
                importance_vars = importance[1] ##the vars defined in importance function
        return np.concatenate(importance_collect, axis= 0), importance_vars
    
    #snapshot of only a subset of outputs from importance function
    def importance_update_subsets(self,cluster_subsets):
        #output data as np for a subset of each cluster
        importance_collect = []
        importance_vars = []
        padding = self.batch_size - cluster_subsets.shape[0]%self.batch_size
        feed = np.concatenate([cluster_subsets, cluster_subsets[-padding-1:-1]])
        list_invar = {
            key: np.split(value, value.shape[0] // self.batch_size)
            for key, value in _DictDatasetMixin._idx_var(self.invar, feed[:,0]).items()
        }
        for i in range(len(next(iter(list_invar.values())))):
            importance = self.importance_measure(
                {key: value[i] for key, value in list_invar.items()}
            )
            importance_collect.append(importance[0]) ##TODO figure this out better
            if not importance_vars:
                importance_vars = importance[1] ##the vars defined in importance function
        return np.concatenate(importance_collect, axis= 0)[0:cluster_subsets.shape[0]], importance_vars
    
    #push latest outputs to tmanager
    def dataset_update(self,importance_collect, importance_vars):
        try:
            output_dataset = _DictDatasetMixin._idx_var({
                k:v.cpu().detach().numpy() for k,v in self.invar.items()
                }, range(0,self.length))
        except:
            print("NO INPUT VARS BEING USED FOR OUTPUT DATASET")
            output_dataset = {}
        
        for k,i in enumerate(importance_vars):
            output_dataset[i] = importance_collect[:,k,None] #combine selected vars for output dataset as dict.
        
        self.TManager.update_dataset_outputs(output_dataset, self.counter)


    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,

        ################### New Args ##################
        warmup: int = 1000,
        initial_graph_vars: List[str] = [],
        graph_vars: List[str] = [],
        SPADE_vars: List[str] = [],
        LOSS_vars: List[str] = [],
        mapping_function: Union[Callable, str] = 'default',
        KNN: int = 10,
        sample_ratio: float = .2,
        sample_bounds: List[float] = [5/100,70/100],
        batch_iterations: int = 1000,
        cluster_size: int = 20,
        coarse_level: int = 1,
        iterations_rebuild: int = 20000,
        local_grid_width: int = 2,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        print(f'local grid width dataset {local_grid_width}')
        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure

        ## NEW
        self.warmup=warmup
        self.initial_graph_vars = initial_graph_vars
        print(f'Initial_Graph_Vars: {self.initial_graph_vars}')
        self.graph_vars = graph_vars
        print(f'Graph_Vars: {self.graph_vars}')
        self.SPADE_vars = SPADE_vars
        print(f'SPADE_vars: {self.SPADE_vars}')
        self.LOSS_vars = LOSS_vars
        print(f'LOSS_vars: {self.LOSS_vars}')
        self.mapping_function = mapping_function
        self.KNN = KNN
        self.sample_ratio = sample_ratio
        self.sample_bounds = sample_bounds

        self.batch_iterations = batch_iterations
        self.cluster_size = cluster_size

        self.coarse_level = coarse_level
        self.iterations_rebuild = iterations_rebuild
        
        self.local_grid_width = local_grid_width
        

        print('IMPORTS')
        from modulus.multithreading.multithreading import MultiProcessSPADE

        self.checkInvar = self.invar.keys()
        self.TManager = MultiProcessSPADE(self.invar, self.batch_size, 
                            self.KNN, self.initial_graph_vars, self.graph_vars,
                            SPADE_vars,LOSS_vars,
                            grid_width = self.local_grid_width,
                            num_eigs = 2, SPADE_GRID = False, 
                            r = 0, sample_ratio = self.sample_ratio,
                            sample_bounds = self.sample_bounds) 
        
        self.counter=0
        
        def iterable_function():        

            self.counter = 0 

            ##Make sure output is different
            if all([i in self.invar.keys() for i in self.graph_vars]):
                print("NOTE: Clustering subsequent graphs on INPUT VARS only, these are static,")

            #check inputs
            if not initial_graph_vars or not graph_vars:
                raise ValueError("Must have input and output vars selected")

            for i in self.warmup_loop(message='GRAPH NOT STARTED, WARMUP SAMPLING'):
                yield i
            print(f'WARMUP DONE')
            
            importance_collect, importance_vars = self.importance_update()

            self.dataset_update(importance_collect, importance_vars)
    
            print('STARTING NEW SPADE CALC')
            print(__name__)
            self.TManager.new_clusters(detail = 'initial')

            #This loop will continue to feed randomized inputs until first SPADE calc is done.
            lstart = time.time()
            for i in self.warmup_loop(indicator_fcn = self.TManager.check_graph_sequential,
                                      threshold = True,
                                      detail='initial',
                                      message='FIRST GRAPH STARTED, WARMUP SAMPLING'):
                yield i
            print(f'FIRST GRAPH DONE')
            
            with open('./logging.txt', 'a+') as f:
                lend = time.time()
                f.write(f'First Graph After Warmup, {lend-lstart}, step:{self.counter}  \n')
                lstart = time.time()

            print(f'LEVEL: {coarse_level}')
            import os
            self.TManager.subset_data_collect(importance_collect, importance_vars, self.batch_size*self.batch_iterations)
            rebuild_reset = 0
            while True: #main loop
                print(f'RUNNING? {self.TManager.running}, counter: {self.counter}')
                #If nothing is being built and T_G is met or was recently met, start a new graph for sparsifying.
                if (self.TManager.running == 0) and (self.counter%self.iterations_rebuild == 0 or rebuild_reset > 0):
                    print(f'ITERATIONS_REBUILD, OUTER LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                    importance_collect, importance_vars = self.importance_update()
                    self.dataset_update(importance_collect, importance_vars)
                    print("Starting new Graph")
                    self.TManager.new_clusters()
                    rebuild_reset = 0
                elif self.TManager.running > 0:
                    print("New graph not finished, resample current.")
                    rebuild_reset = 2
                while True:
                    if rebuild_reset == 1:
                        print(f'ITERATIONS_REBUILD, INNER LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                        break
                    elif rebuild_reset == 2:
                        print(f'GRAPH WAS NOT READY!{self.counter}, continuing')
                        rebuild_reset = 1
                    idx = self.TManager.batch_receiver(self.batch_iterations)
                    backup_batch = False
                    if idx.ndim > 1:
                        idx = idx[:,0].astype(np.int64)
                        self.TManager.subset_refresh(self.importance_update_subsets,self.batch_size*self.batch_iterations)
                    else:
                        backup_batch = True
                    for i in range(0,idx.shape[0]//self.batch_size): 
                        self.counter += 1
                        if self.counter%500 == 0:
                            print(f"running?{self.TManager.running}")
                        if self.counter%1000 == 0 and self.TManager.running:
                            if self.TManager.check_graph_sequential(self.counter):
                                print("New graph done")
                                with open('./logging.txt', 'a+') as f:
                                    lend = time.time()
                                    f.write(f'New Graph, {lend-lstart}, step:{self.counter}\n')
                                    self.TManager.subset_data_collect(importance_collect, importance_vars, self.batch_size*self.batch_iterations)
                                    lstart = time.time()
                                    print('Loaded new graph')
                                break
                        if self.counter%self.iterations_rebuild == 0:
                            print(f'ITERATIONS_REBUILD, INNER FOR LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                            rebuild_reset = 1
                            break
                        # gather invar, outvar, and lambda weighting
                        batch_range = range(i*self.batch_size, i*self.batch_size + self.batch_size)
                        
                        invar = _DictDatasetMixin._idx_var(self.invar, idx[batch_range]) ##formerly [:,0]
                        outvar = _DictDatasetMixin._idx_var(self.outvar, idx[batch_range])
                        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx[batch_range])

                        # return and count up
                        assert self.invar.keys() == self.checkInvar
                        yield (invar, outvar, lambda_weighting)

                    if not backup_batch:
                        break

        self.iterable_function = iterable_function

        def save_dataset_importance_batch(invar):
            
            out = []
            list_importance = []
            list_invar = { ### variable: list of batches 
                key: np.split(value, value.shape[0] // self.batch_size)
                for key, value in invar.items()
            }
            for i in range(len(next(iter(list_invar.values())))):
                importance = self.importance_measure(
                    {key: value[i] for key, value in list_invar.items()}
                )
                list_importance.append(importance)
            cluster_values = np.concatenate(list_importance, axis= 0)

            out = cluster_values
            return out
            
        self.save_batch_function = save_dataset_importance_batch

    def __iter__(self):
        yield from self.iterable_function()


class ListIntegralDataset(_DictDatasetMixin, Dataset):
    """
    A map-style dataset for a finite set of integral training examples.
    """

    auto_collation = True

    def __init__(
        self,
        list_invar: List[Dict[str, np.array]],
        list_outvar: List[Dict[str, np.array]],
        list_lambda_weighting: List[Dict[str, np.array]] = None,
    ):
        if list_lambda_weighting is None:
            list_lambda_weighting = []
            for outvar in list_outvar:
                list_lambda_weighting.append(
                    {key: np.ones_like(x) for key, x in outvar.items()}
                )

        invar = _stack_list_numpy_dict(list_invar)
        outvar = _stack_list_numpy_dict(list_outvar)
        lambda_weighting = _stack_list_numpy_dict(list_lambda_weighting)

        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length

    def save_dataset(self, filename):
        for idx in range(self.length):
            var_to_polyvtk(
                filename + "_" + str(idx).zfill(5),
                _DictDatasetMixin._idx_var(self.invar, idx),
            )

class No_SPADE_GraphImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):
    """
    An infinitely iterable dataset that implements SGM without SPADE; older version used to run the LDC examples.
    """

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,

        ################### New Args ##################
        warmup: int = 1000,
        initial_graph_vars: List[str] = [],
        graph_vars: List[str] = ["x", "y", "z"],
        mapping_function: Union[Callable, str] = 'default',
        KNN: int = 10,
        sample_ratio: float = .2,
        divisions: int = 1,
        coarse_level: int = 1,
        R: int = 1,
        sparsification: int = 0,
        epochs_resample: int = 45,
        local_grid_width: int = 2,
        iterations_rebuild: int = 20000
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        print(f'local grid width dataset {local_grid_width}')
        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure

        ## NEW
        self.warmup=warmup
        self.initial_graph_vars = initial_graph_vars
        print(f'Initial_Graph_Vars: {self.initial_graph_vars}')
        self.graph_vars = graph_vars
        print(f'Graph_Vars: {self.graph_vars}')
        self.mapping_function = mapping_function
        self.KNN = KNN
        self.sample_ratio = sample_ratio
        self.divisions = divisions
        self.coarse_level = coarse_level
        self.R = R
        self.sparsification = sparsification
        self.epochs_resample = epochs_resample
        self.local_grid_width = local_grid_width
        self.iterations_rebuild = iterations_rebuild

        print('IMPORTS')
        import modulus.external.coarsen_utils as GraphUtils
        from modulus.multithreading.multithreading_no_SPADE import MultiProcessHyperEF



        self.TManager = MultiProcessHyperEF(self.invar, self.batch_size, 
                            self.KNN, self.sparsification,
                            self.coarse_level, self.R, 
                            self.initial_graph_vars, self.graph_vars, 
                            grid_width = local_grid_width) 
        

        def iterable_function():        
            initial_idx = range(0,self.length)

            counter = 0 #for warmup periods and resampling timers

            initial_inputs = _DictDatasetMixin._idx_var({
                k:v.cpu().detach().numpy() for k,v in self.invar.items() if k in self.initial_graph_vars + self.graph_vars
                }, initial_idx)
            initial_importance_vals = []
            initial_importance_vars = []

            #input-only graph
            if initial_graph_vars:
                initial_clusters = None
                lstart = time.time()
                print('STARTING INITIAL NEW')
                print(__name__)
                self.TManager.new_graph(detail = 'initial')

            #Initial graph begins immediately on the input data; subsequent graphs can be delayed by an initial warmup
            while counter < self.warmup:
                print(counter)
                if self.TManager.next_laplacian():
                    print("New graph done")
                    with open('./logging.txt', 'a+') as f:
                        lend = time.time()
                        f.write(f'New Graph Input-Only, {lend-lstart}, step:{counter} \n')
                        lstart = time.time()
                    initial_clusters = self.TManager.combined_graph
                    print('Loaded new graph')
                if self.TManager.combined_graph == None:
                    warmup_idx = np.split(np.random.permutation(initial_idx), self.length // self.batch_size)
                    for i in warmup_idx:
                        if counter%1000 == 0:
                            if initial_graph_vars:
                                print('Making input-only graph. WARMUP SAMPLING.')
                            else:
                                print('Graph Not Started. WARMUP SAMPLING.')
                        invar = _DictDatasetMixin._idx_var(self.invar, i)
                        outvar = _DictDatasetMixin._idx_var(self.outvar, i)
                        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, i)
                        counter += 1
                        yield (invar, outvar, lambda_weighting)
                else:
                    print(f'WARMUP SAMPLING from {initial_graph_vars}')
                    cluster_subsets = GraphUtils.getMappingCore(initial_clusters, total = 1, 
                            repetition = True, shuffle = False, 
                            sample = self.sample_ratio)
                    print('Re-calculating Importance')
                    list_importance = []
                    padding = self.batch_size - cluster_subsets.shape[0]%self.batch_size
                    feed = np.concatenate([cluster_subsets, cluster_subsets[-padding-1:-1]])
                    list_invar = {
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in _DictDatasetMixin._idx_var(self.invar, feed[:,0]).items()
                    }
                    for i in range(len(next(iter(list_invar.values())))):
                        importance = self.importance_measure(
                            {key: value[i] for key, value in list_invar.items()}
                        )
                        list_importance.append(importance[0]) ##TODO figure this out better
                    cluster_values = np.concatenate(list_importance, axis= 0)[0:-padding]
                    idx = GraphUtils.getMappingCore_weighted_fast_modulus(initial_clusters, cluster_subsets,
                                                    cluster_values,
                                                    sMin = 5/100, sMax = 70/100,
                                                    total = self.epochs_resample, shuffle = self.shuffle,
                                                    avg = self.sample_ratio > 0)

                    idx = idx.astype(np.int64)
                    for i in range(0,idx.shape[0]//self.batch_size):
                        counter += 1
                        # gather invar, outvar, and lambda weighting
                        batch_range = range(i*self.batch_size, i*self.batch_size + self.batch_size)
                        invar = _DictDatasetMixin._idx_var(self.invar, idx[:,0][batch_range])
                        outvar = _DictDatasetMixin._idx_var(self.outvar, idx[:,0][batch_range])
                        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx[:,0][batch_range])
                        yield (invar, outvar, lambda_weighting)

            if not all([i in self.invar.keys() for i in self.graph_vars]):
                print(f'Calculating current outputs for clustering...{self.graph_vars}')
                list_invar = {
                    key: np.split(value, value.shape[0] // self.batch_size)
                    for key, value in _DictDatasetMixin._idx_var(self.invar, initial_idx).items()
                }
                for i in range(len(next(iter(list_invar.values())))):
                    importance = self.importance_measure(
                        {key: value[i] for key, value in list_invar.items()}
                    )
                    initial_importance_vals.append(importance[0])
                    if not initial_importance_vars:
                        initial_importance_vars = importance[1]
                initial_importance_vals = np.concatenate(initial_importance_vals, axis= 0)
                for k,i in enumerate(initial_importance_vars):
                    if i in self.graph_vars: #not all importance vars may be used in clustering
                        initial_inputs[i] = initial_importance_vals[:,k,None]
                self.TManager.update_dataset(initial_inputs)
            else:
                print('Inputs Only Clustering')

            print('STARTING_NEW')
            print(__name__)
            self.TManager.new_graph()

            if self.TManager.combined_graph == None:
                lstart = time.time()
            while self.TManager.combined_graph == None:
                print(counter)
                warmup_idx = np.split(np.random.permutation(initial_idx), self.length // self.batch_size)
                for i in warmup_idx:
                    if counter%1000 == 0:
                        print('WARMUP SAMPLING')
                        print(f'next_laplacian: {self.TManager.next_laplacian()}')
                        print(f'tmanager.comb_graph: {type(self.TManager.combined_graph)}')
                    invar = _DictDatasetMixin._idx_var(self.invar, i)
                    outvar = _DictDatasetMixin._idx_var(self.outvar, i)
                    lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, i)
                    counter += 1
                    yield (invar, outvar, lambda_weighting)
            
            with open('./logging.txt', 'a+') as f:
                lend = time.time()
                f.write(f'First Graph After Warmup, {lend-lstart}, step:{counter}  \n')
                lstart = time.time()

            print(f'LEVEL: {coarse_level}')
            initial_clusters = self.TManager.combined_graph
            
            #main loop
            while True:
                #re-calculate full dataset importance scores.
                if (not self.TManager.running) and counter%self.iterations_rebuild == 0:
                    if not all([i in self.invar.keys() for i in self.graph_vars]):
                        print(f'Calculating current outputs for clustering...{self.graph_vars}')
                        initial_importance_vals = []
                        list_invar = {
                            key: np.split(value, value.shape[0] // self.batch_size)
                            for key, value in _DictDatasetMixin._idx_var(self.invar, initial_idx).items()
                        }
                        for i in range(len(next(iter(list_invar.values())))):
                            importance = self.importance_measure(
                                {key: value[i] for key, value in list_invar.items()}
                            )
                            initial_importance_vals.append(importance[0])
                            if not initial_importance_vars:
                                initial_importance_vars = importance[1]
                        initial_importance_vals = np.concatenate(initial_importance_vals, axis= 0)
                        for k,i in enumerate(initial_importance_vars):
                            if i in self.graph_vars: #not all importance vars may be used in clustering
                                initial_inputs[i] = initial_importance_vals[:,k,None]
                        self.TManager.update_dataset(initial_inputs)
                    else:
                        print('Inputs Only Clustering')
                    print("Starting new Graph")
                    self.TManager.new_graph()
                elif self.TManager.running:
                    print("New graph not finished, resample current.")
                cluster_subsets = GraphUtils.getMappingCore(initial_clusters, total = 1, repetition = True, shuffle = False, sample = self.sample_ratio)
                print('Re-calculating Importance')


                #inner loop
                while True:
                    #re-calculate subset importance
                    list_importance = []
                    padding = self.batch_size - cluster_subsets.shape[0]%self.batch_size
                    feed = np.concatenate([cluster_subsets, cluster_subsets[-padding-1:-1]])
                    list_invar = {
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in _DictDatasetMixin._idx_var(self.invar, feed[:,0]).items()
                    }
                    for i in range(len(next(iter(list_invar.values())))):
                        importance = self.importance_measure(
                            {key: value[i] for key, value in list_invar.items()}
                        )
                        list_importance.append(importance[0])
                    cluster_values = np.concatenate(list_importance, axis= 0)[0:-padding]
                    #Makes several mini-epochs for sampling from latest subset values.
                    idx = GraphUtils.getMappingCore_weighted_fast_modulus(initial_clusters, cluster_subsets,
                                                    cluster_values,
                                                    sMin = 5/100, sMax = 70/100,
                                                    total = self.epochs_resample, shuffle = self.shuffle,
                                                    avg = self.sample_ratio > 0)
                    
                    idx = idx.astype(np.int64)
                    for i in range(0,idx.shape[0]//self.batch_size): 
                        counter += 1
                        if counter%self.iterations_rebuild == 0: #start new graph if T_G met.
                            break
                        if counter%500 == 0:
                            print(f"running?{self.TManager.running}")
                        if counter%1000 == 0 and self.TManager.running: #periodically check status of any running graph
                            if self.TManager.next_laplacian():
                                print("New graph done")
                                with open('./logging.txt', 'a+') as f:
                                    lend = time.time()
                                    f.write(f'New Graph, {lend-lstart}, step:{counter}\n')
                                    initial_clusters = self.TManager.combined_graph
                                    lstart = time.time()
                                    print('Loaded new graph')
                                break
                        batch_range = range(i*self.batch_size, i*self.batch_size + self.batch_size)
                        
                        invar = _DictDatasetMixin._idx_var(self.invar, idx[:,0][batch_range])
                        outvar = _DictDatasetMixin._idx_var(self.outvar, idx[:,0][batch_range])
                        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx[:,0][batch_range])

                        # return and count up
                        yield (invar, outvar, lambda_weighting)
                    break

        self.iterable_function = iterable_function

        def save_dataset_importance_batch(invar):
            
            out = []
            list_importance = []
            list_invar = { 
                key: np.split(value, value.shape[0] // self.batch_size)
                for key, value in invar.items()
            }
            for i in range(len(next(iter(list_invar.values())))):
                importance = self.importance_measure(
                    {key: value[i] for key, value in list_invar.items()}
                )
                list_importance.append(importance)
            cluster_values = np.concatenate(list_importance, axis= 0)

            out = cluster_values
            return out
            
        self.save_batch_function = save_dataset_importance_batch

    def __iter__(self):
        yield from self.iterable_function()

class ContinuousIntegralIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of integral training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        batch_size: int,
        lambda_weighting_fn: Callable = None,
        param_ranges_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }
        if param_ranges_fn is None:
            param_ranges_fn = lambda: {} 
        self.param_ranges_fn = param_ranges_fn

        self.batch_size = batch_size


        def iterable_function():
            while True:
                list_invar = []
                list_outvar = []
                list_lambda_weighting = []
                for _ in range(self.batch_size):
                    param_range = self.param_ranges_fn()
                    list_invar.append(self.invar_fn(param_range))
                    if (
                        not param_range
                    ):  
                        param_range = {"_": next(iter(list_invar[-1].values()))[0:1]}

                    list_outvar.append(self.outvar_fn(param_range))
                    list_lambda_weighting.append(
                        self.lambda_weighting_fn(param_range, list_outvar[-1])
                    )
                invar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_invar))
                outvar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_outvar))
                lambda_weighting = Dataset._to_tensor_dict(
                    _stack_list_numpy_dict(list_lambda_weighting)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        return list(invar.keys())

    @property
    def outvar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictVariationalDataset(Dataset):
    """
    A map-style dataset for a finite set of variational training examples.
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.outvar_names = outvar_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return invar

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.outvar_names)

    def save_dataset(self, filename):
        for i, invar in self.invar.items():
            var_to_polyvtk(invar, filename + "_" + str(i))


def _stack_list_numpy_dict(list_var):
    var = {}
    for key in list_var[0].keys():
        var[key] = np.stack([v[key] for v in list_var], axis=0)
    return var
