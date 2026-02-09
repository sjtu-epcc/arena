#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
Core implementations of cell class and related functions to unify cluster scheduling 
and parallelism search. 
"""

from typing import (
    Sequence, Any, List, Tuple, Dict, Union,
)
from uuid import uuid1
from itertools import (
    permutations, count)
import pickle
import numpy as np

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxpr.utils import (
    HardwareConfigs, StageShape, PipelinePlan, CellConfigs, is_power_of)


class Cell:
    """ 
    The class of cell to represent a subspace of the complex ```cluster scheduleing x hybrid parallelism''' 
    searching space with (1) specified GPU resources and (2) determined number of pipeline stages. 
    """

    name_id = count()
    
    def __init__(
        self,
        job_id: str, 
        model_name: str,
        hardware_cfgs: HardwareConfigs, 
        cell_cfgs: CellConfigs = None,
        num_stages: int = 1,
        uuid: str = None
    ) -> None:
        """ Initialize a cell object. """

        self.uuid = uuid or Cell._next_id()
        self.job_id = job_id
        self.model_name = model_name

        # Configs for parallelism determination
        self.cell_cfgs = cell_cfgs or CellConfigs()

        # Properties
        self.num_hosts = hardware_cfgs.num_nodes
        self.num_devices_per_host = hardware_cfgs.num_devices_per_node
        self.num_gpus = self.num_hosts * self.num_devices_per_host
        self.gpu_type = hardware_cfgs.gpu_type
        self.num_stages = num_stages        # Only forward stages

        # Pipeline partition plan
        self.pipeline_plan = None
        self.pipeline_strategy = None
        self._preset_physical_shapes = None

        # Parallel plans
        # Describe how each pipeline stage is parallelized, each plan is in the format of 
        # `[(data para degree, model para degree), ...]`, or `[StageShape, ...]`. Can be preset when the cell profile is 
        # disabled (i.e., use one parallelism  specified by user) or auto-generated with cell profile.
        self.parallel_plans = []
        self._parallel_plan_hints = None
        self._estimated_optimal_plan = None

        # Cell performance
        self.perf_lookup_table = {}

        self._comm_time_table_cache = {}
    
    ################################
    #         Class Methods        #
    ################################
    
    @classmethod
    def _next_id(cls):
        return f"cell_{next(cls.name_id)}"
    
    @classmethod
    def gen_hashkey_with_parallelism(
        self, 
        submesh_logical_shapes: Sequence[Sequence[int]],
    ) -> str:
        """ 
        Generate hash key with the given parallelism to store HLO intermediate results. 
        The format is: "{dp1}_{mp1}__{dp2}_{mp2}" (for 2 stages).
        """

        hashkey = ""
        for _i, _submesh in enumerate(submesh_logical_shapes):
            _suffix = f"__{_submesh[0]}_{_submesh[1]}" \
                if _i > 0 else f"{_submesh[0]}_{_submesh[1]}"
            hashkey += _suffix
        
        return hashkey
    
    @classmethod
    def gen_hashkey_with_partition_plan(
        self, 
        partition_plan: List[List[int]] = None,
        plan_hashkey: str = None,
        decode: bool = False,
    ) -> str:
        """ 
        Generate hash key with the specified partition plan. 
        The format is: f"{start_layer_idx_1}_{end_layer_idx_1}__{start_layer_idx_2}_{end_layer_idx_2}" 
        (if the layers are clustered into 2 stages).
        """

        if decode:
            plan = []
            for substr in plan_hashkey.split("__"):
                s_idx, e_idx = tuple([int(_v) for _v in substr.split("_")])
                plan.append([_i for _i in range(s_idx, e_idx + 1, 1)])
            return plan

        hashkey = ""
        for (i, stage) in enumerate(partition_plan):
            suffix = f"__{stage[0]}_{stage[-1]}" if i > 0 else f"{stage[0]}_{stage[-1]}"
            hashkey += suffix
        
        return hashkey
    
    @classmethod
    def gen_hashkey_with_stage_shape(
        self, 
        stage_shape: StageShape = None,
        shape_hashkey: str = None,
        decode: bool = False,
    ) -> Union[str, StageShape]:
        """
        Generate hash key with the specified stage logical shape in the format of `(dp_rank, tp_rank)`. 
        The format is: f"{dp_rank}_{tp_rank}".
        """
        if decode:
            if "_" not in shape_hashkey:
                # Mixed
                return StageShape(type="mixed", stage_shape=None, layer_shape=None)
            else:
                # Single
                return StageShape(
                    type="single", stage_shape=tuple([int(_c) for _c in shape_hashkey.split("_")]), layer_shape=None,
                )

        if stage_shape.type == "single":
            return f"{stage_shape.stage_shape[0]}_{stage_shape.stage_shape[1]}"
        elif stage_shape.type == "mixed":
            return stage_shape.type
        else:
            raise ValueError(f"Invalid type of stage shape: {stage_shape.type}")

    ################################
    #    Status Related Methods    #
    ################################
    
    def is_pipeline_partitioned(self) -> bool:
        """ 
        Whether the job allocated with this cell has been 
        pipeline-partitioned. 
        """
        return (self.pipeline_plan is not None)
    
    def is_parallel_enumerated(self) -> bool:
        """ Whether the parallel plans are enumerated. """
        return (len(self.parallel_plans) > 0)
    
    def is_profiled(self) -> bool:
        """ 
        Whether the job allocated with this cell has been 
        profiled. 
        """
        return (len(list(self.perf_lookup_table.keys())) > 0)
    
    def num_parallel_prof(self) -> int:
        """ Get the number of profiled parallelisms. """
        return len(list(self.perf_lookup_table.keys()))
    
    ################################
    #   Parallel Related Methods   #
    ################################

    def set_parallel_plan_hints(
        self,
        stage_all_logical_shapes: Dict[int, Dict[str, int]],
    ) -> None:
        """ 
        Set the hints for cell's exploration and profiling on parallel plan. 
        
        Args:
         - `stage_all_logical_shapes`: The communication of all stages and all candidate logical shapes 
                                       (i.e., parallel plans) under cell's pipeline plan with the format of:
                                       `stage_idx -> shape_hashkey -> (aggregated GPU fraction, allocated GPU num, 
                                       computation loads, intra per-stage comm, additional comm, inter-stages comm)`.
        """
        self._parallel_plan_hints = stage_all_logical_shapes

        print(f"[I] Stage all logical shapes: {self._parallel_plan_hints}")

    def preset_parallel_plans(
        self, 
        parallel_plans: Sequence[Any],
    ) -> None:
        """ Preset parallel plans (i.e., list of submesh logical shapes) when the cell profile is disabled. """
        
        for i in range(len(parallel_plans)):
            if not isinstance(parallel_plans[i][0], StageShape):
                # Convert to StageShape object. Default to single shape.
                parallel_plans[i] = [
                    StageShape(type="single", stage_shape=(_s[0], _s[1]), layer_shape=None) for _s in parallel_plans[i]
                ]

            # Check whether DP degree <= local batch size
            local_batch_size = int(os.environ.get("CRIUS_LOCAL_BATCH_SIZE", "1"))
            for s in parallel_plans[i]:
                assert s.stage_shape[0] <= local_batch_size, \
                    f"DP degree ({s.stage_shape[0]}) should be no more than local batch size ({local_batch_size})."

        self.parallel_plans = parallel_plans
    
    def enum_parallel_plans(
        self, 
        strategy: str = "auto",
    ) -> None:
        """ 
        Instead of profile user-specified parallelism, enumerate and profile candidate parallel plans 
        (i.e., submesh logical shapes) with the specified resource quota and number of pipeline stages. 

        Args:
         - `strategy`: The strategy of cell profiling. Options: ["minimal", "uniform", "auto"]. If set to 
                       "minimal", only enumerate vanilla data and tensor parallelism; if set to "uniform", 
                       enumerate all uniform (symmetric) parallel plans with constraint of #stages, including 
                       hybrid ones. If set to "auto", use the parallel plan hints generated by the pipeline
                       planner to select the theoratically best-performing plan as the target parallel plan.
        """

        assert self.is_pipeline_partitioned(), \
            "Cell needs to be pipeline-partitioned before " + \
            "enumerating parallel plans."
        
        if strategy == "minimal":
            self._enum_parallel_plans_minimal()
        elif strategy == "uniform":
            self._enum_parallel_plans_uniform()
        elif strategy == "auto":
            self._enum_parallel_plans_auto()
        else:
            raise ValueError(f"Invalid parallel enumerating strategy: {strategy}")
        
        # TODO(chunyu): This is a post sanity check to ensure: DP degree <= local batch size.
        local_batch_size = int(os.environ.get("CRIUS_LOCAL_BATCH_SIZE", "1"))
        for plan in self.parallel_plans:
            for s in plan:
                assert s.stage_shape[0] <= local_batch_size, \
                    f"DP degree ({s.stage_shape[0]}) should be no more than local batch size ({local_batch_size})."


    def _enum_parallel_plans_uniform(self) -> None:
        """ 
        Enumerate all uniform (symmetric) parallel plans with the constraint of stage num if the generated
        GPU sharding is uniform. Otherwise, use "minimal" option.
        """

        assert all([
            _quota == self.pipeline_plan.gpu_sharding[0] 
                for _quota in self.pipeline_plan.gpu_sharding
        ]) and is_power_of(2, len(self.pipeline_plan.gpu_sharding)), \
            f"The generation of uniform parallel plans is " + \
            f"not supported when the generated GPU sharding " + \
            f"({self.pipeline_plan.gpu_sharding}) is not uniform. " + \
            f"Use 'minimal' option instead."
        
        self.parallel_plans = _enum_parallel_with_gpu_sharding(
            self.pipeline_plan.gpu_sharding)
    
    def _enum_parallel_plans_minimal(self) -> None:
        """
        Enumerate parallel plans where only vanilla data parallelism or vanilla tensor parallelism is 
        enabled with the constraint of stage num. If #stages = #GPUs, only enumerate one parallel plan 
        that performs vanilla pipeline parallelism.
        """

        if self.num_stages < self.num_gpus:
            self.parallel_plans = [
                # Data parallelism only
                [(_num_gpus, 1) for _num_gpus in self.pipeline_plan.gpu_sharding],
                # Tensor parallelism only
                [(1, _num_gpus) for _num_gpus in self.pipeline_plan.gpu_sharding]
            ]
        elif self.num_stages == self.num_gpus:
            self.parallel_plans = [[(1, 1) for _ in range(self.num_stages)]]
        else:
            raise RuntimeError(
                f"The number of stages ({self.num_stages}) must be no more than " + 
                f"the number of GPUs ({self.num_gpus}).")
        
        # Convert to StageShape object.
        self.parallel_plans = [
            [StageShape(type="single", stage_shape=(_s[0], _s[1]), layer_shape=None) for _s in _plan]
                for _plan in self.parallel_plans
        ]
    
    def _enum_parallel_plans_auto(
        self,
        _gamma: float = 1.1,
    ) -> None:
        """
        Sequentually construct the theoratically best-performing parallel plan based on the theoratical intra-stage
        communication of varying stages and varying logical shapes (i.e., per-stage parallel plan).

        Similar to the sequential shape inferring in `../pipeline/planner/_infer_min_intra_stage_comm_fixed_plan()`.
        """

        if os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none") != "none":
            # Forcibly overwrite stage shapes
            shape_hashkeys = os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none").split("::")[1].split("__")
            parallel_plan = [
                Cell.gen_hashkey_with_stage_shape(shape_hashkey=_k, decode=True)
                    for _k in shape_hashkeys
            ]
            self.parallel_plans.append(parallel_plan)
            return 
        
        if self._estimated_optimal_plan is not None:
            # Use the plan estimated from non-dominated set
            plan_hashkey = self._estimated_optimal_plan.split("::")[0]
            fw_stage_layer_ids = Cell.gen_hashkey_with_partition_plan(plan_hashkey=plan_hashkey, decode=True)
            num_layers = fw_stage_layer_ids[-1][-1] + 1
            bw_stage_layer_ids = [
                [
                    2 * num_layers - 1 - _i 
                        for _i in reversed(_layer_ids)
                ] for _layer_ids in reversed(fw_stage_layer_ids)
            ]

            assert (self.pipeline_plan.stage_layer_ids == fw_stage_layer_ids + bw_stage_layer_ids), \
                "Mismatched determined pipeline partition of the cell and the estimated optimal plan."

        # Greedily estimate the optimal stage shapes by minimizing per-stage intra-stage communication
        assert self._parallel_plan_hints is not None, \
            f"The parallel plan hints of the cell is not set from the pipeline planner."
        
        parallel_plan = []
        last_stage_shape = None
        plan_hashkey = Cell.gen_hashkey_with_partition_plan(self.pipeline_plan.stage_layer_ids[:self.num_stages])
        for stage_idx in self._parallel_plan_hints[plan_hashkey]:
            # For each stage
            min_comm, best_shape = 1e12, None
            for shape_hashkey in self._parallel_plan_hints[plan_hashkey][stage_idx]:
                # For each shape
                shape = Cell.gen_hashkey_with_stage_shape(shape_hashkey=shape_hashkey, decode=True)
                (_, _, _, per_stage_comm, addn_comm, _) = self._parallel_plan_hints[plan_hashkey][stage_idx][shape_hashkey]
                # Estimated total comm for this stage
                stage_total_comm = (per_stage_comm + addn_comm) \
                    if (per_stage_comm is not None and addn_comm is not None) else 1e12
                
                if stage_idx > 0 and shape != last_stage_shape:
                    stage_total_comm *= _gamma

                if stage_total_comm < min_comm:
                    best_shape = shape
                    min_comm = stage_total_comm
            
            parallel_plan.append(best_shape)
            last_stage_shape = best_shape
        
        # # TODO(chunyu): Manually specify for dummy test.
        # parallel_plan = [
        #     StageShape(type="single", stage_shape=(2, 1), layer_shape=None),
        #     StageShape(type="single", stage_shape=(2, 1), layer_shape=None),
        # ]

        self.parallel_plans.append(parallel_plan)
    
    def preset_submesh_physical_shapes(
        self, 
        shapes: Sequence[Sequence[int]],
    ) -> None:
        """ 
        Preset submesh physical shapes (i.e., gpu allocation)
        to skip auto-generation based on gpu sharding. Mostly
        used in uniform pipeline scenarios.
        """
        self._preset_physical_shapes = shapes

    def gen_submesh_physical_shapes(self) -> List[Tuple[int]]:
        """ 
        Generate formulated submesh physical shapes based on
        GPU sharding and allocated resources.
        """

        if self._preset_physical_shapes is not None:
            # Return preset shapes
            print(f"[I] Pipeline partition mode: {self.pipeline_strategy} " + 
              f"| Preset physical mesh shapes: {self._preset_physical_shapes}")
            return self._preset_physical_shapes
        
        assert self.is_pipeline_partitioned(), \
            "This cell is not pipeline-partitioned yet."
        shapes =  gen_submesh_physical_shapes(self.pipeline_plan.gpu_sharding,
                                              self.num_hosts,
                                              self.num_devices_per_host)
        self._preset_physical_shapes = shapes     
        print(f"[I] Pipeline partition mode: {self.pipeline_strategy} " + 
              f"| Cell-generated physical mesh shapes: {shapes}")   
    
        return shapes
    
    ################################
    #   Profile Related Methods    #
    ################################
    
    def update_perf(
        self,
        hashkey: str, 
        e2e_iter_time: float,
    ) -> None:
        """ 
        Update profiling results with the given hashkey (related to parallelism) and performance data. 

        Args: 
         - `hashkey`: Hashkey generated by `Cell.gen_hashkey_with_parallelism()` as the key in the 
                      performance lookup table.
         - `e2e_iter_time`: End-to-end iteration time to be recorded.
        """

        assert hashkey not in self.perf_lookup_table, \
            f"Hash crash occurs with key '{hashkey}'."
        self.perf_lookup_table[hashkey] = e2e_iter_time
    
    def last_perf_hashkey(self) -> str:
        """ 
        Get the last hashkey in the performance lookup table. 
        """
        return list(self.perf_lookup_table.keys())[-1]

    ################################
    #    Tuning Related Methods    #
    ################################

    def gen_pareto_plan_set_for_tuning(
        self,
        num_micro_batches: int,
    ) -> List[str]:
        """
        Generate a set of Pareto optimal plans on (1) pipeline partition and (2) intra-stage 
        parallelization as the searching space of job tuning.

        The optimizing objectives include: (1) Minimize end-to-end communication constructed based on
        the common pipeline structure; (2) Minimize the gap (i.e., L2 norm) between the aggregated
        GPU fraction of each stage and its allocated GPU quota.

        We borrow the idea of Strength Pareto Evolutionary Algorithm (SPEA) algorithm to implement 
        a lightweight candidate selection method for constructing the set of Pareto optimal plans, 
        since in our scenario the value range of plan is discrete and finite. 

        Workflow:

         - Initialize a non-dominated set P that stores all the searched pareto optimal plans.

         - Traverse each plan with varying pipeline partition and intra-stage parallelization, for each
           plan i, compare it with all plans in P. If plan i is dominated (both objectives are worse than 
           one plan in P) by any plan in P, drop i; otherwise, add plan i into P and drop all plans in P
           that are dominated by plan i.
        
         - If the size of P is larger than `max_set_size`, use the SPEA Clustering technique to reduce
           the size of P. Specifically, we use Edit Distance (Levenshtein Distance) of layer-stage-id vector 
           to represent the 'distance' between two plans. For example, in a 4-layer and 2-stage case,
           the distance between plan `[0, 0, 1, 1]` and plan `[0, 0, 0, 1]` is 1. 

           We iteratively select two plans in P that lead to the minimal distance, drop the one with higher
           L2 norm on compute-imbalance (if same, drop the one with higher e2e communication), until the 
           size of P is reduced to no more than `max_set_size`.

         - Return the non-dominated set P as the Pareto optimal plan set.

        Args:
         - `num_micro_batches`: Number of micro batches.
        """

        # Non-dominated set where each elite plan is represented as:
        #   f"{plan_hashkey}::{shape_hashkey_stage_1}__{shape_hashkey_stage_2}"
        non_dominated_set = []

        for plan_hashkey in self._parallel_plan_hints:
            plan = Cell.gen_hashkey_with_partition_plan(plan_hashkey=plan_hashkey, decode=True)
            
            if self.num_stages == 1:
                # Only one stage
                for shape_hashkey in self._parallel_plan_hints[plan_hashkey][0]:
                    non_dominated_set = self._update_non_dominated_set(
                        plan_hashkey, [shape_hashkey], non_dominated_set, num_micro_batches)
            
            # Queue for storing intermediate partial plans in plan enumeration
            plan_queue = []
            for shape_hashkey in self._parallel_plan_hints[plan_hashkey][0]:
                plan_queue.append([shape_hashkey])

            for stage_idx in range(1, self.num_stages, 1):
                # Loop for all partial plans with the length of `stage_idx`
                while len(plan_queue) > 0 and len(plan_queue[0]) == stage_idx:
                    # Pop the element in the queue head
                    partial_plan = plan_queue.pop(0)
                    
                    for shape_hashkey in self._parallel_plan_hints[plan_hashkey][stage_idx]:
                        if self._parallel_plan_hints[plan_hashkey][stage_idx][shape_hashkey][0] is not None:
                            # Stage is executable
                            new_partial_plan = partial_plan + [shape_hashkey]

                            if len(new_partial_plan) == self.num_stages:
                                # Complete plan construction and update non-dominated set
                                if self._invalid_plan_shape_hashkey(plan_hashkey, new_partial_plan):
                                    # Skip invalid 
                                    continue

                                non_dominated_set = self._update_non_dominated_set(
                                    plan_hashkey, new_partial_plan, non_dominated_set, num_micro_batches,
                                )
                            else:
                                # Queue in
                                plan_queue.append(new_partial_plan)
        
        def __cal_l2_norm(plan_shape_hashkey: str):
            """
            Calculate the l2 normalization on the GPU fraction of all stages (i.e., the similarity of the
            aggregated GPU fraction and the actually allocated quota). 
            """
            plan_hashkey = plan_shape_hashkey.split("::")[0]
            shape_hashkeys = plan_shape_hashkey.split("::")[1].split("__")
            stage_gpu_fractions, stage_alloc_gpu_nums = [], []

            for (stage_idx, shape_hashkey) in enumerate(shape_hashkeys):
                (sum_gpu_fraction, 
                 alloc_gpu_num, 
                 _, _, _, _) = self._parallel_plan_hints[plan_hashkey][stage_idx][shape_hashkey]
                stage_gpu_fractions.append(sum_gpu_fraction)
                stage_alloc_gpu_nums.append(alloc_gpu_num)
            
            return np.sqrt(np.sum((np.array(stage_gpu_fractions) - np.array(stage_alloc_gpu_nums)) ** 2))

        # Sort by l2 norm
        non_dominated_set = sorted(non_dominated_set, key=__cal_l2_norm)
        # Select the most compute-balance one as the estimated optimal plan
        if len(non_dominated_set) > 0:
            self._estimated_optimal_plan = non_dominated_set[0]

        print("[TMP] Plans in the non-dominated set:")

        for plan_shape_hashkey in non_dominated_set:
            strs = plan_shape_hashkey.split("::")
            plan = Cell.gen_hashkey_with_partition_plan(plan_hashkey=strs[0], decode=True)
            stage_shapes = [
                Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                    for _s in strs[1].split("__")
            ]

            print(plan_shape_hashkey)
            print(plan)
            print(stage_shapes)

            target_l2_norm, target_e2e_comm = self._query_hints(
                plan, stage_shapes, num_micro_batches,
            )
            print(target_l2_norm)
            print(target_e2e_comm)
            print("")
        
        # exit(0)
        
        return non_dominated_set
    
    def _invalid_plan_shape_hashkey(
        self,
        plan_hashkey: str,
        shape_hashkeys: List[str],
    ) -> bool:
        """ Whether this plan shape hashkey is invalid. """

        last_shape_hashkey, last_alloc_gpu_num = None, None
        for (stage_idx, shape_hashkey) in enumerate(shape_hashkeys):
            (_, alloc_gpu_num, _,
             _, _, _) = self._parallel_plan_hints[plan_hashkey][stage_idx][shape_hashkey]
            
            if stage_idx == 0:
                last_shape_hashkey = shape_hashkey
                last_alloc_gpu_num = alloc_gpu_num
                continue
            
            if (self.cell_cfgs.only_symmetric_sharding and alloc_gpu_num != last_alloc_gpu_num):
                return True
            
            if (self.cell_cfgs.only_universal_shape and shape_hashkey != last_shape_hashkey):
                return True
            
            if (
                self.cell_cfgs.universal_shape_stage_num > 0 and 
                stage_idx < self.cell_cfgs.universal_shape_stage_num and
                shape_hashkey != last_shape_hashkey
            ):
                return True
            
            if (
                self.cell_cfgs.max_universal_shape_num > 0 and
                (shape_hashkeys.count(max(set(shape_hashkeys), key=shape_hashkeys.count)) > 
                    self.cell_cfgs.max_universal_shape_num)
            ):
                return True

            last_shape_hashkey = shape_hashkey
            last_alloc_gpu_num = alloc_gpu_num

        return False
    
    def _update_non_dominated_set(
        self,
        new_plan_hashkey: str,
        new_shape_hashkeys: List[str],
        non_dominated_set: List[str],
        num_micro_batches: int,
    ) -> List[str]:
        """ 
        Update non-dominated set based on the new plan.

        Args:
         - `new_plan_hashkey`: The encoded hashkey of the new partition plan.
         - `new_shape_hashkeys`: List of encoded hashkeys to represent the logical shape of each stage.
         - `non_dominated_set`: Non-dominated set that stores Pareto optimal plans.
        """

        new_plan = Cell.gen_hashkey_with_partition_plan(plan_hashkey=new_plan_hashkey, decode=True)
        new_stage_shapes = [
            Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) for _s in new_shape_hashkeys
        ]
        to_drop_hashkeys = []

        # Inspect elite plans in non-dominated set
        for plan_shape_hashkey in non_dominated_set:
            strs = plan_shape_hashkey.split("::")
            plan = Cell.gen_hashkey_with_partition_plan(plan_hashkey=strs[0], decode=True)
            stage_shapes = [
                Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                    for _s in strs[1].split("__")
            ]
            
            if self._dominate(plan, stage_shapes, new_plan, new_stage_shapes, num_micro_batches):
                # Case 1. This new plan is dominated by an existing elite plan, drop the new plan
                assert len(to_drop_hashkeys) == 0, \
                    f"In non-dominated plan set, there cannot be any plan that is dominated by the other plan."
                return non_dominated_set
            
            if self._dominate(new_plan, new_stage_shapes, plan, stage_shapes, num_micro_batches):
                # Case 2. This new plan can dominate this existing elite plan, drop the existing one
                to_drop_hashkeys.append(plan_shape_hashkey)
        
        # Drop elite plans dominated by the new plan
        for plan_shape_hashkey in to_drop_hashkeys:
            non_dominated_set.remove(plan_shape_hashkey)
        
        # Add the new plan into non-dominated set
        multi_shapes_hashkey = "__".join(new_shape_hashkeys)
        non_dominated_set.append(f"{new_plan_hashkey}::{multi_shapes_hashkey}")

        if len(non_dominated_set) > self.cell_cfgs.max_plan_set_size:
            # Use clustering to reduce the size of the non-dominated set
            assert len(non_dominated_set) == self.cell_cfgs.max_plan_set_size + 1, \
                f"Size of the non-dominated set can only be one element larger than the maximum set size."
            non_dominated_set = self._cluster_one_plan(non_dominated_set, num_micro_batches)
        
        return non_dominated_set

    def _dominate(
        self,
        target_plan: List[List[int]],
        target_stage_shapes: List[StageShape],
        other_plan: List[List[int]],
        other_stage_shapes: List[StageShape],
        num_micro_batches: int,
        _beta: float = 0.1,
    ) -> bool:
        """ 
        Target plan & shape can dominate other plan & shape, i.e., with lower L2 norm for evaluating 
        compute-imbalance and e2e communication. 
        """

        target_l2_norm, target_e2e_comm = self._query_hints(
            target_plan, target_stage_shapes, num_micro_batches,
        )
        other_l2_norm, other_e2e_comm = self._query_hints(
            other_plan, other_stage_shapes, num_micro_batches,
        )

        if self._cannot_determine_comm_dominating_relation(
            target_plan, target_stage_shapes, other_plan, other_stage_shapes,
        ):
            # Cannot determine the dominating relationship of two plans w.r.t. communication.
            # In this case, if the l2 norm of the target plan is significantly smaller than
            # that of the other plan, we think it can dominate.
            return (target_l2_norm <= other_l2_norm * _beta)

        return (target_l2_norm <= other_l2_norm and target_e2e_comm <= other_e2e_comm)

    def _query_hints(
        self,
        partition_plan: List[List[int]],
        stage_shapes: List[StageShape],
        num_micro_batches: int,
        _gamma: float = 1.1,
        _lambda: float = 0.5,
    ) -> Tuple[float, float]:
        """ 
        Query hints of parallel plans to get two objective values of the given plan (&shape): 
        
         - The gap (L2 norm) between the aggregated GPU fraction of each stage and its allocated GPU num.

         - The end-to-end communication constructed based on the common pipeline structure, including both
           intra-stage and inter-stages communication.
        
        Args:
         - `partition_plan`: List of layer ids in each stage.
         - `stage_shapes`: Lisf of logical shape of each stage.
        """

        plan_hashkey = Cell.gen_hashkey_with_partition_plan(partition_plan)
        raw_gpu_sharding, norm_gpu_sharding, inter_stages_comms = [], [], []
        max_per_stage_comm = -1e12
        sum_addn_comm = 0
        last_stage_shape = None

        for (stage_idx, shape) in enumerate(stage_shapes):
            shape_hashkey = Cell.gen_hashkey_with_stage_shape(shape)
            # Metrics
            (sum_gpu_fraction, 
             alloc_gpu_num, 
             _,
             per_stage_comm, 
             addn_comm,
             inter_stage_comm) = self._parallel_plan_hints[plan_hashkey][stage_idx][shape_hashkey]
            
            if sum_gpu_fraction is None:
                # Infeasible parallelism
                return 1e12, 1e12
            
            if stage_idx > 0 and shape.stage_shape != last_stage_shape.stage_shape:
                # Penalty of cross-stages resharding
                per_stage_comm *= _gamma
                addn_comm *= _gamma

            # Record
            raw_gpu_sharding.append(sum_gpu_fraction)
            norm_gpu_sharding.append(alloc_gpu_num)
            max_per_stage_comm = max(per_stage_comm, max_per_stage_comm)
            sum_addn_comm += addn_comm
            inter_stages_comms.append(inter_stage_comm)
            last_stage_shape = shape
        
        # Objective 1: l2 norm for evaluating compute-imbalance
        l2_norm = np.sqrt(np.sum((np.array(raw_gpu_sharding) - np.array(norm_gpu_sharding)) ** 2))

        # Objective 2: e2e communication with intra-stage and inter-stages scenarios
        intra_stage_obj_val = max_per_stage_comm + sum_addn_comm
        inter_stage_obj_val = np.sum(inter_stages_comms) + _lambda * np.std(inter_stages_comms[:-1]) \
            if len(inter_stages_comms) > 1 else 0
        e2e_comm = inter_stage_obj_val + (num_micro_batches - 1) * intra_stage_obj_val
        
        return l2_norm, e2e_comm
    
    def _cannot_determine_comm_dominating_relation(
        self,
        target_plan: List[List[int]],
        target_stage_shapes: List[StageShape],
        other_plan: List[List[int]],
        other_stage_shapes: List[StageShape],
        _eps: float = 0.5,
    ) -> bool:
        """ 
        Cannot determine the dominating relationship of two plans w.r.t. communication.

        Use (1) communication amount and offline profiled communication data, and (2) stage flops 
        and roofline attainable performance to compare in quite coarse-grained level. 
        If the communication is significantly (i.e., 10x) smaller than the computation, then we think 
        that we cannot determine the intra-stage parallelism of this stage.
        """

        # target_plan = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        # other_plan = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

        # target_stage_shapes = [
        #     StageShape(type="single", stage_shape=(2, 1), layer_shape=None),
        #     StageShape(type="single", stage_shape=(2, 1), layer_shape=None),
        # ]

        # other_stage_shapes = [
        #     StageShape(type="single", stage_shape=(1, 2), layer_shape=None),
        #     StageShape(type="single", stage_shape=(2, 1), layer_shape=None),
        # ]

        # print("")
        # print(f"Target plan: {target_plan} | Stage shapes: {target_stage_shapes}")
        # print(f"Other plan: {other_plan} | Stage shapes: {other_stage_shapes}")

        def __comm_significant(comm_size: int, stage_shape: Tuple[int], comp_load: float):
            """ Whether the communication is significant compared to computation load. """
            if np.prod(stage_shape) == 1:
                # Only one gpu, must be not significant
                return False
            
            estimated_comm_lat = estimate_comm_time(
                comm_size, self.gpu_type, stage_shape, self._comm_time_table_cache,
            )
            
            # print("Estimated communication latency:", estimated_comm_lat)
            # print("Computation load:", comp_load)
            # print("")
            
            if estimated_comm_lat < comp_load * _eps:
                return False
            return True

        num_stages = len(target_plan)
        # Plan hashkey
        target_plan_hashkey = Cell.gen_hashkey_with_partition_plan(target_plan)
        other_plan_hashkey = Cell.gen_hashkey_with_partition_plan(other_plan)
        # Physical shapes
        target_physical_shapes = gen_submesh_physical_shapes(
            gpu_sharding=[np.prod(_s.stage_shape) for _s in target_stage_shapes],
            num_hosts=self.num_hosts,
            num_devices_per_host=self.num_devices_per_host,
        )
        other_physical_shapes = gen_submesh_physical_shapes(
            gpu_sharding=[np.prod(_s.stage_shape) for _s in other_stage_shapes],
            num_hosts=self.num_hosts,
            num_devices_per_host=self.num_devices_per_host,
        )

        for stage_idx in range(num_stages):
            # Determine whether for this stage, the intra-stage communication of both target and other 
            # plan is significant. If not so, we cannot determine the dominating relationship on 
            # intra-stage communication.

            # Stage shapes
            target_physical_shape = target_physical_shapes[stage_idx]
            other_physical_shape = other_physical_shapes[stage_idx]
            if np.prod(target_physical_shape) != np.prod(other_physical_shape):
                # If the stage gpu num of target and other plan are different, we cannot determine the
                # dominating relationship on intra-stage communication.
                return True

            # Target plan
            target_shape_hashkey = Cell.gen_hashkey_with_stage_shape(target_stage_shapes[stage_idx])
            (_, _, 
             comp_load,
             per_stage_comm, 
             addn_comm, _) = self._parallel_plan_hints[target_plan_hashkey][stage_idx][target_shape_hashkey]
            target_comm_significant = __comm_significant(
                per_stage_comm + addn_comm, target_physical_shape, comp_load,
            )

            # Other plan
            other_shape_hashkey = Cell.gen_hashkey_with_stage_shape(other_stage_shapes[stage_idx])
            (_, _, 
             comp_load,
             per_stage_comm, 
             addn_comm, _) = self._parallel_plan_hints[other_plan_hashkey][stage_idx][other_shape_hashkey]
            other_comm_significant = __comm_significant(
                per_stage_comm + addn_comm, other_physical_shape, comp_load,
            )
        
            # print(f"Target plan communication of stage {stage_idx} is significant: {target_comm_significant}")
            # print(f"Other plan communication of stage {stage_idx} is significant: {other_comm_significant}")
            # print("")

            if target_comm_significant or other_comm_significant:
                # Can determine
                return False
        
        return True

    def _cluster_one_plan(
        self,
        non_dominated_set: List[str],
        num_micro_batches: int,
    ) -> List[str]:
        """ Cluster two elite plans with the minimal edit distance. """

        # Look for plan pair with the minimal edit distance
        min_dist, max_l2_norm_sum, max_e2e_comm_sum, plan_shape_hashkey_pair = 1e12, -1e12, -1e12, None
        for i in range(0, len(non_dominated_set) - 1, 1):
            for j in range(i + 1, len(non_dominated_set), 1):
                # Plan i
                plan_i = Cell.gen_hashkey_with_partition_plan(
                    plan_hashkey=non_dominated_set[i].split("::")[0], decode=True,
                )
                stage_shapes_i = [
                    Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                        for _s in non_dominated_set[i].split("::")[1].split("__")
                ]
                l2_norm_i, e2e_comm_i = self._query_hints(plan_i, stage_shapes_i, num_micro_batches)
                
                # Plan j
                plan_j = Cell.gen_hashkey_with_partition_plan(
                    plan_hashkey=non_dominated_set[j].split("::")[0], decode=True,
                )
                stage_shapes_j = [
                    Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                        for _s in non_dominated_set[j].split("::")[1].split("__")
                ]
                l2_norm_j, e2e_comm_j = self._query_hints(plan_j, stage_shapes_j, num_micro_batches)

                dist = self._cal_edit_distance(plan_i, plan_j)
                if (
                    # Smaller edit distance
                    dist < min_dist or 
                    # Larger sum of l2 norm
                    (dist == min_dist and 
                     l2_norm_i + l2_norm_j > max_l2_norm_sum) or
                     # Larger sum of e2e comm
                    (dist == min_dist and 
                     l2_norm_i + l2_norm_j == max_l2_norm_sum and 
                     e2e_comm_i + e2e_comm_j > max_e2e_comm_sum)
                ):
                    min_dist = dist
                    max_l2_norm_sum = l2_norm_i + l2_norm_j
                    max_e2e_comm_sum = e2e_comm_i + e2e_comm_j
                    plan_shape_hashkey_pair = (non_dominated_set[i], non_dominated_set[j])

        # Plan 1
        plan_1 = Cell.gen_hashkey_with_partition_plan(
            plan_hashkey=plan_shape_hashkey_pair[0].split("::")[0], decode=True,
        )
        stage_shapes_1 = [
            Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                for _s in plan_shape_hashkey_pair[0].split("::")[1].split("__")
        ]
        l2_norm_1, e2e_comm_1 = self._query_hints(plan_1, stage_shapes_1, num_micro_batches)

        # Plan 2
        plan_2 = Cell.gen_hashkey_with_partition_plan(
            plan_hashkey=plan_shape_hashkey_pair[1].split("::")[0], decode=True,
        )
        stage_shapes_2 = [
            Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True) 
                for _s in plan_shape_hashkey_pair[1].split("::")[1].split("__")
        ]
        l2_norm_2, e2e_comm_2 = self._query_hints(plan_2, stage_shapes_2, num_micro_batches)

        if (
            l2_norm_1 > l2_norm_2 or
            (l2_norm_1 == l2_norm_2 and e2e_comm_1 >= e2e_comm_2)
        ):
            # Drop plan 1
            non_dominated_set.remove(plan_shape_hashkey_pair[0])
        else:
            # Drop plan 2
            non_dominated_set.remove(plan_shape_hashkey_pair[1])

        return non_dominated_set

    def _cal_edit_distance(
        self,
        plan_self: List[List[int]],
        plan_other: List[List[int]],
    ) -> int:
        """ Calculate the edit distance (levenshtein distance) between two partition plans. """

        def __get_stage_id(layer_id: int, plan: List[List[int]]):
            """ Get the stage id of the specified layer. """
            for (i, stage) in enumerate(plan):
                if layer_id in stage:
                    return i
            return None
        
        assert len(plan_self) == len(plan_other), \
            f"Mismatched number of stages between two plans."
        # Flatten
        num_layers = plan_self[-1][-1] + 1
        layer_stage_ids_self = [__get_stage_id(_i, plan_self) for _i in range(num_layers)]
        layer_stage_ids_other = [__get_stage_id(_i, plan_other) for _i in range(num_layers)]

        return sum(1 for i, j in zip(layer_stage_ids_self, layer_stage_ids_other) if i != j)


def estimate_comm_time(
    comm_size: int,
    gpu_type: str,
    mesh_shape: Tuple[int],
    comm_time_table_cache: Dict[str, Dict[str, List[Any]]],
) -> float:
    """ 
    Estimate coarse-grained communication time cost based on the offline 
    profiled communication data. 
    """

    comm_time_table = None
    cfg_hashkey = f"{gpu_type}_{mesh_shape[0]}_{mesh_shape[1]}"
    if cfg_hashkey in comm_time_table_cache:
        # Load cache
        comm_time_table = comm_time_table_cache[cfg_hashkey]
    else:
        # Read files
        if os.environ.get("USE_IB_COMM_DATA", "false") == "true":
            # Use comm data profiled with infiniband
            comm_file_name = f"{mesh_shape[0]}_{gpu_type}_{mesh_shape[0]}_n_{mesh_shape[1]}_d_ib.pkl"
        else:
            # Use comm data without infiniband
            comm_file_name = f"{mesh_shape[0]}_{gpu_type}_{mesh_shape[0]}_n_{mesh_shape[1]}_d.pkl"
        comm_data_pth = os.path.join(os.environ.get("COMM_LOG_PATH"), comm_file_name)
        with open(comm_data_pth, "rb") as f:
            comm_time_table = pickle.load(f)
        # Cache
        comm_time_table_cache[cfg_hashkey] = comm_time_table

    op_type = "all-reduce"  # Estimate as all-reduce primitive
    replica_groups = [
        [
            _i + mesh_shape[1] * _j 
                for _i in range(mesh_shape[1])
        ] for _j in range(mesh_shape[0])
    ]
    comm_hashkey = str((op_type, replica_groups))
    last_comm_size, last_comm_time = -1, -1

    if comm_size <= np.prod(comm_time_table[comm_hashkey][0][0]):
        return comm_time_table[comm_hashkey][0][2]
    
    for i, (shape, _, comm_time, _) in enumerate(comm_time_table[comm_hashkey]):  
        _comm_size = np.prod(shape)
        if _comm_size >= comm_size:
            # Linear interpolate
            _ratio = (comm_size - last_comm_size) / (_comm_size - last_comm_size)
            return last_comm_time + (comm_time - last_comm_time) * _ratio
        
        last_comm_size, last_comm_time = _comm_size, comm_time

    # Exceed max profiled comm size
    max_profiled_comm_size = np.prod(comm_time_table[comm_hashkey][-1][0])
    # Decompose communication size into profiled scope
    max_profiled_cnt = comm_size // max_profiled_comm_size
    rest_comm_time = estimate_comm_time(
        comm_size % max_profiled_comm_size, gpu_type, mesh_shape, comm_time_table_cache,
    )
    max_comm_time = comm_time_table[comm_hashkey][-1][2]

    return max_profiled_cnt * max_comm_time + rest_comm_time


def _enum_parallel_with_gpu_sharding(gpu_sharding: Sequence[int]) -> List[List[StageShape]]:
    """ 
    Enumerate all uniform (symmetric) submesh logical shapes 
    (i.e., parallelisms) when the specified GPU sharding is uniform. 
    """

    parallel_plans = []
    log_nd = int(np.log2(np.sum(gpu_sharding)))
    p_d = int(np.log2(len(gpu_sharding)))    
    
    for d_d in range(0, log_nd - p_d + 1, 1):
        m_d = log_nd - p_d - d_d
        parallel_plans.append([
            (pow(2, d_d), pow(2, m_d)) for _ in range(pow(2, p_d))
        ])
    
    return [
        [StageShape(type="single", stage_shape=(_s[0], _s[1]), layer_shape=None) for _s in _plan]
            for _plan in parallel_plans
    ]


def gen_submesh_physical_shapes(
    gpu_sharding: Sequence[int], 
    num_hosts: int, 
    num_devices_per_host: int,
) -> List[Tuple[int]]:
    """ 
    Generate submesh physical shapes based on allocated devices and gpu sharding.
    Format: `[(1, 2) (2 GPUs on host 1), (2, 4) (8 GPUs on host 2 and 3)]`.

    Args: 
     - `gpu_sharding`: List of allocated GPU num for each stage.
     - `num_hosts`: Number of hosts.
     - `num_devices_per_host`: Number of devices on each host.
    """

    cur_num_hosts = num_hosts
    cur_num_devices_per_host = num_devices_per_host
    submesh_physical_shapes = []

    for _quota in gpu_sharding:
        if _quota >= num_devices_per_host:
            # Inter-hosts mesh
            submesh_physical_shapes.append(
                (_quota // num_devices_per_host, num_devices_per_host)
            )
            cur_num_hosts -= _quota // num_devices_per_host
        elif _quota <= cur_num_devices_per_host:
            # Intra-host mesh
            submesh_physical_shapes.append((1, _quota))
            cur_num_devices_per_host -= _quota
            if cur_num_devices_per_host == 0:
                cur_num_devices_per_host = num_devices_per_host
                cur_num_hosts -= 1
        else:
            # Partial inter-hosts mesh
            assert _quota == 2 * cur_num_devices_per_host, \
                f"In partial inter-hosts mesh case, stage GPU num {_quota} must be 2 x " + \
                f"remained bubbles num ({cur_num_devices_per_host}) of the partially allocated host."
            submesh_physical_shapes.append((2, cur_num_devices_per_host))
            cur_num_hosts -= 1
            cur_num_devices_per_host = num_devices_per_host - cur_num_devices_per_host
    
    assert cur_num_hosts == 0, \
        f"Uncleared tmp num_hosts ({cur_num_hosts}), which should be 0."
    
    return submesh_physical_shapes
