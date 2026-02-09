#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to transform flax-defined DNN model into the Jaxpr intermediate representation. """

from typing import Sequence, Any, Dict, Optional, Set
import numpy as np
from functools import partial
from itertools import permutations
import time
import jax
import jax.numpy as jnp
from jax import linear_util as lu
from jax.core import (
    Jaxpr, ClosedJaxpr, JaxprEqn, Var, DropVar, gensym, subjaxprs)
from jax.api_util import (argnums_partial, donation_vector,
                          flatten_fun_nokwargs, rebase_donate_argnums)
from jax.tree_util import (tree_flatten, tree_unflatten, PyTreeDef)
from jax.experimental.maps import FrozenDict
from jax._src.util import HashableFunction
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
import alpa
# from alpa.device_mesh import VirtualPhysicalMesh
from alpa.util import (
    OrderedSet, GradFuncTransformContext, trace_jaxpr_with_micro_batch, 
    auto_static_argnums, auto_donate_argnums, abstractify_with_aval)
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call,
    create_donation_mapping, generate_computations_from_modules,
    generate_sharded_xla_computations,
    generate_sharded_xla_computations_arguments, get_donatable_intermediate,
    mark_missing_vars_in_backward_computation_pipeline_marks, pipeline_dce,
    slice_closed_jaxpr_by_full_pipeline_marks, split_donate_invars,
    XlaShardedPipelineComputation)
from alpa.pipeline_parallel.stage_construction import (
    get_stage_outvars, get_sliced_virtual_submeshes)
from alpa.pipeline_parallel.layer_construction import (
    ManualLayerOption, AutoLayerOption, automatic_layer_construction)
from alpa.pipeline_parallel.compile_executable import (
    split_and_process_layers, slice_apply_grad_for_stage_construction, 
    _rewrite_global_outvars_post_concate)
from alpa.pipeline_parallel.apply_grad import (
    apply_grad_get_mean, compute_grad_to_accumulate_grad,
    process_apply_gradient, split_compute_grad_and_apply_grad)
from alpa.pipeline_parallel.schedules import (
    gen_dependency_with_stages, PipeDreamFlush, GpipeSchedule)
from alpa.pipeline_parallel.layer_stats import eqn_flops

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell.cell import Cell
from pipeline.planner import PipelinePlanner
from crius_worker.jax.trainer.wide_resnet_trainer import WideResNetTrainer
from crius_worker.jax.utils import (TrainState, create_learning_rate_fn)
from jaxpr.preprocess import (
    prepare_flax_model, get_stage_layer_ids_uniform)
from jaxpr.utils import (
    ParallelMethod, HardwareConfigs, InputConfigs)
from jaxpr.testing import get_dummy_input_cfgs
from jaxpr.custom_mesh import VirtualPhysicalMesh


def decode_fun_args(fun, *args):
    """ 
    Decode arguments of the parallelized function by flattening PyTree arguments.
    --------------------------------------------------------------
    Modified from: alpa/api/ParallelizedFunc/_decode_args_and_get_executable() 
    """
    # The same as the static_argnums argument of jax.jit. 
    # If it is "auto", alpa uses heuristic rules to infer this.
    static_argnums = "auto"
    # The same as the donate_argnums argument of jax.jit. 
    # If it is "auto", alpa uses heuristic rules to infer this.
    donate_argnums = "auto"
    # The indices of arguments that are the data batch. 
    # This information is used to split the original data batch into micro batches 
    # to perform gradient accumulation or pipeline parallelism. 
    # Alpa assumes the 0-th dimension of the tensor is the batch dimension.
    batch_argnums = (1,)

    kwargs = {}

    f = lu.wrap_init(fun)

    # Deal with static arguments and extract dynamic arguments
    if static_argnums == "auto":
        static_argnums = auto_static_argnums(args)

    if static_argnums:
        dyn_argnums = [
            i for i in range(len(args)) if i not in static_argnums
        ]
        # Freeze static dict to make it hashable
        frozen_args = []
        for i, arg in enumerate(args):
            if i in static_argnums and isinstance(arg, dict):
                frozen_args.append(FrozenDict(arg))
            else:
                frozen_args.append(arg)
        f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
    else:
        dyn_args = args
    
    # Flatten pytree arguments
    args_flat, in_tree = tree_flatten(dyn_args)
    f, out_tree = flatten_fun_nokwargs(f, in_tree)
    # pylint: disable=unnecessary-lambda
    out_tree_hashable = HashableFunction(lambda: out_tree(), closure=None)

    # Deal with donate argnums
    if donate_argnums == "auto":
        donate_argnums = auto_donate_argnums(args)

    donate_tuple = rebase_donate_argnums(donate_argnums, static_argnums)
    if donate_tuple:
        donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
    else:
        donated_invars = (False,) * len(args_flat)

    # Deal with batch argnums
    batch_tuple = rebase_donate_argnums(batch_argnums, static_argnums)
    batch_invars = donation_vector(batch_tuple, dyn_args, kwargs)

    # A map object, should be accessed with '*' as a sequence of abstract values
    abstract_args = map(abstractify_with_aval, args_flat)
    # avals = list(abstract_args)

    return f, in_tree, out_tree_hashable, static_argnums, donated_invars, batch_invars, *abstract_args


def compile_and_make_jaxpr(
    train_step_fn, 
    state, 
    micro_batch, 
    batch, 
    num_micro_batches, 
    layer_num, 
    use_remat=True, 
    naive_trace=False,
):
    """ Compile and make the jaxpr representation. """

    # Layer option
    remat_mode = "coarse_grained_remat" if use_remat else "none"
    layer_option = AutoLayerOption(layer_num=layer_num, remat_mode=remat_mode)
    # layer_option = train_step_fn.method.layer_option
    # with GradFuncTransformContext(ManualLayerOption(use_remat).transform):
    with GradFuncTransformContext(layer_option.transform):
        if naive_trace:
            batch_invars = None
            # Trace the function with a micro batch to get the jaxpr.
            closed_jaxpr, output_tree = jax.make_jaxpr(
                fun=train_step_fn, return_shape=True,
            )(state, micro_batch)
            # Trace again with a full batch. The full batch is used to derive 
            # the reduction operator across micro batches (e.g., addition, concatenation).
            full_batch_closed_jaxpr, full_batch_output_tree = jax.make_jaxpr(
                fun=train_step_fn, return_shape=True,
            )(state, batch)
        else:
            # Decode the naive train step function
            (wrapped_fun, in_tree, out_tree_hashable, 
             static_argnums, donated_invars, 
             batch_invars, *avals) = decode_fun_args(train_step_fn, state, batch)
            
            # Clean stores for the next call
            for store in wrapped_fun.stores:
                if store:
                    store.reset()
            batch_invars = list(batch_invars)
            for idx, aval in enumerate(avals):
                if len(aval.shape) == 0:
                    batch_invars[idx] = False
            batch_invars = tuple(batch_invars)

            # Trace the function with a micro batch to get the jaxpr.
            closed_jaxpr, micro_batch_size = trace_jaxpr_with_micro_batch(
                wrapped_fun, batch_invars, num_micro_batches, avals,
            )
            # Trace again with a full batch.
            # The full batch is used to derive the reduction operator across
            # micro batches (e.g., addition, concatenation).
            full_batch_output_tree = None
            if num_micro_batches > 1:
                for store in wrapped_fun.stores:
                    if store:
                        store.reset()
                full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
                    wrapped_fun, batch_invars, 1, avals,
                )
            else:
                full_batch_closed_jaxpr = None
            # Output tree
            output_tree = out_tree_hashable()

    # Invars
    num_params = len(closed_jaxpr.jaxpr.invars) - len(batch)
    donated_invars = [True] * num_params + [False] * len(batch)
    
    return (closed_jaxpr, output_tree, full_batch_closed_jaxpr, full_batch_output_tree, 
            donated_invars, batch_invars, layer_option, in_tree, static_argnums)


def generate_virtual_physical_mesh(cell: Cell):
    """ 
    Generate a customized virtual physical mesh based on the given 
    cell, which is used for compilation. 
    """

    # Distribute each partition/replica to a single host
    num_hosts = cell.num_hosts
    num_devices_per_host = cell.num_devices_per_host
    host_ids = np.arange(num_hosts)
    host_info = [dict() for _ in range(num_hosts)]
    
    return VirtualPhysicalMesh(
        host_ids=host_ids, 
        host_info=host_info, 
        num_devices_per_host=num_devices_per_host,
    )


def transform_to_jaxpr_and_cluster_stages(
    trainer: Any, 
    train_step_fn: Any, 
    state: Any, 
    micro_batch: Any, 
    batch: Any, 
    cell: Cell, 
    pipeline_planner: PipelinePlanner,
    use_remat: bool = False, 
    parallel_method: ParallelMethod = None, 
    virtual_mesh: VirtualPhysicalMesh = None
):
    """ 
    Transform a Jax-defined model into Jaxpr Intermediate Representation (IR) 
    in layer-style, then cluster them into stages. 
    """

    print(f"[I] Compiling model and making jaxpr...")

    # Compile
    layer_num = trainer.num_pipeline_layers
    num_micro_batches = trainer.num_micro_batches
    (closed_jaxpr, output_tree, full_batch_closed_jaxpr, 
     full_batch_output_tree, donated_invars, batch_invars, 
     layer_option, in_tree, static_argnums) = compile_and_make_jaxpr(
         train_step_fn, state, micro_batch, batch, num_micro_batches, layer_num, use_remat,
     )

    gensym_func = gensym([closed_jaxpr.jaxpr])
    global_invars = closed_jaxpr.jaxpr.invars
    
    # Split and process the input jaxpr into pipelined forward and 
    # backward + grad accumulation style.
    (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
    microbatch_bound, reduction_vector, post_microbatch_bound,
    accumulator_mapping, acc_grad_invars,
    acc_grad_outvars) = split_and_process_layers(
        closed_jaxpr, full_batch_closed_jaxpr, trainer.num_micro_batches, False, gensym_func,
    )

    # Split apply_grad jaxpr into layers
    (jax_apply_layers, _) = slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, global_invars, 
         global_outvars, donated_invars, accumulator_mapping, gensym_func, False,
     )
    
    print(f"[I] Jaxpr has been transformed into {len(jax_pipeline_layers) + len(jax_apply_layers)} layers.")
    
    # Construct Jaxpr stages
    strategy = "auto" if os.environ.get("CRIUS_ENABLE_CELL_PROFILE", "false") == "true" else "uniform"
    jax_pipeline_stages = pipeline_planner.cluster_layers_into_stages(
            strategy, jax_pipeline_layers, jax_apply_layers, cell, 
            parallel_method.submesh_logical_shapes, layer_num, 
            acc_grad_outvars, accumulator_mapping,
        )
    
    # Submesh physical shapes
    submesh_physical_shapes = cell.gen_submesh_physical_shapes()

    num_meshes = len(jax_pipeline_stages) // 2
    assert num_meshes == len(cell.pipeline_plan.gpu_sharding), \
        f"Mismatched stage num {num_meshes} with gpu sharding " + \
        f"num {len(cell.pipeline_plan.gpu_sharding)}"
    
    # Generate sliced virtual physical meshes
    sliced_virtual_meshes = get_sliced_virtual_submeshes(virtual_mesh, submesh_physical_shapes)
    # Handle the process of applying gradient and creating donation. 
    num_devices = [_vmesh.num_devices for _vmesh in sliced_virtual_meshes]
    (sliced_apply_grad_stages, apply_grad_placement,
     global_outvars, allreduce_groups) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, jax_pipeline_stages, 
         cell.pipeline_plan.stage_to_mesh, gensym_func, num_meshes, 
         global_invars, global_outvars, donated_invars, False, num_devices)
    jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages

    donation_mapping = create_donation_mapping(
        accumulator_mapping, donated_invars, global_invars, global_outvars,
    )
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages, gensym_func,
    )
    global_outvars, concat_vars_mapping = _rewrite_global_outvars_post_concate(
        global_outvars, reduction_vector, microbatch_bound, post_microbatch_bound, gensym_func,
    )

    # Generate pipeline schedule and placement
    dependency = gen_dependency_with_stages(jax_pipeline_stages,
                                            sliced_apply_grad_stages)

    # Apply a Gpipe-style pipeline schedule
    schedule = GpipeSchedule(dependency=dependency, meshes=sliced_virtual_meshes, 
                             apply_grad_placement=apply_grad_placement,
                             num_batch=num_micro_batches)
    
    return (jax_all_stages, sliced_virtual_meshes, schedule, num_meshes, accumulator_mapping, 
            global_invars, num_micro_batches, donate_invars_dict, acc_grad_outvars, global_outvars, 
            submesh_physical_shapes, )


def prepare_model_and_transform_jaxpr(
    input_cfgs: InputConfigs, 
    cell: Cell,
    pipeline_planner: PipelinePlanner,
    use_remat: bool = False, 
    virtual_mesh: VirtualPhysicalMesh = None,
):
    """
    Prepare Jax-based model, transform into Jaxpr IR and cluster layers into pipeline stages. 

    Args:
     - `input_cfgs`: Input configurations.
     - `cell`: Cell to be pipeline-partitioned and profiled.
     - `pipeline_planner`: Pipeline planner to perform stage partition and GPU allocation 
                           the the cell.
     - `use_remat`: Use rematerialization (i.e., gradient chechpointiing).
     - `virtual_mesh`: The virtual mesh object used in virtual device slicing for stages.
    """

    _time_marker = time.time()
    
    # Prepare model
    trainer, train_step_fn, state, micro_batch, batch = prepare_flax_model(input_cfgs, enable_alpa=False)
    
    print(f"[I] Model has been loaded ({time.time() - _time_marker} s), begin Jaxpr transformation...")
    print("")
    
    # Transform to Jaxpr IR and cluster layers into stages
    (jax_all_stages, sliced_virtual_meshes, 
     schedule, num_meshes, accumulator_mapping, 
     global_invars, num_micro_batches, donate_invars_dict, 
     acc_grad_outvars, global_outvars, 
     submesh_physical_shapes) = transform_to_jaxpr_and_cluster_stages(
         trainer, train_step_fn, state, micro_batch, batch, cell, pipeline_planner,
         use_remat, input_cfgs.parallel_method, virtual_mesh,
     )
    
    # Update global invars to handle some invar missing. Global invars
    # is obtained by closed_jaxpr.jaxpr.invars and is only used for getting
    # donatable invars of each stage 
    # (alpa/pipeline_parallel/computation/get_donatable_intermediate()).
    # Since only vars that are both outvar and invar can be donated, updating
    # global invars here won't affect the results.
    # TODO(chunyu): Why this occurs?

    # Outvars of all stages
    _all_outvars = OrderedSet()
    for _stage in jax_all_stages:
        for _outvar in _stage.outvars:
            if _outvar not in _all_outvars:
                _all_outvars.add(_outvar)
    for _outvar in global_outvars:
        if _outvar not in _all_outvars:
            _all_outvars.add(_outvar)
    
    # Mark all missing invars (i.e., not outvars of any stage) as 
    # global invars. 
    for _i, _stage in enumerate(jax_all_stages):
        for _invar in _stage.invars:
            if (_invar not in _all_outvars and 
                _invar not in global_invars):
                # Neither an outvar of certain stage  nor an global invar, 
                # treat as global invar.
                # print(f"[WARN] Invar {_invar} is missing and thus treated as global invar.")
                global_invars.append(_invar)
    
    return (jax_all_stages, sliced_virtual_meshes, schedule, num_meshes, accumulator_mapping, 
            global_invars, num_micro_batches, donate_invars_dict, acc_grad_outvars, 
            submesh_physical_shapes)
