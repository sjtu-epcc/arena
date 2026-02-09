"""
#############################
#        TUTORIALS          #
#############################

[Distributed Training with Both Shard and Pipeline Parallelism]: https://alpa.ai/tutorials/pipeshard_parallelism.html

"""

#####################################################################
# Step 1. Import Libraries and Initialize Environment               #
#####################################################################

import alpa
from alpa.testing import assert_allclose

import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
import time

print("[I] Dependencies load completed.")


#####################################################################
# Step 2. Connect to a Ray Cluster                                  #
#####################################################################

# Alternatively, you can use the command `ray.init(address="auto")` to connect to an existing ray cluster.
# Note: `alpa.init(cluster="ray")` uses the gpus resources of the whole ray cluster. To configure Alpa to only use a subset of gpu resources, one can
# specific the number of nodes and number of gpus per node.
# For example, only run 2 gpus when 8 gpus are available: `alpa.init('ray', num_devices_per_node=2, num_nodes=1)`.
ray.init(address="auto")
alpa.init(cluster="ray")

# NOTE: Get ray node ip address
# @ray.remote
# def f():
#     time.sleep(0.01)
#     return ray._private.services.get_node_ip_address()

# print(ray.get(f.remote()))


#####################################################################
# Step 3. Train an MLP on a Single Device                           #
#####################################################################


# In this tutorial, we use a toy dataset to train an MLP model. Specifically, we use the model to fit the function: y = Wx + b.


def trainMLPOnSingleDev():
    # Note that now this model is being executed on CPU because we force the driver process to use the CPU.
    print("[I] Try to train MLP model on a single device as ground truth.")

    dim = 2048
    batch_size = 2048

    class MLPModel(nn.Module):
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
            return x


    # Define the training step
    def train_step(state, batch):

        def loss_func(params):
            out = model.apply(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        grads = jax.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state


    # Generate ground truth W and b
    rngkey = jax.random.PRNGKey(0)
    k1, k2 = random.split(rngkey)
    W = random.normal(k1, (dim, dim))
    b = random.normal(k2, (dim,))

    # Generate the training data
    ksample, knoise = random.split(k1)
    x = random.normal(ksample, (batch_size, dim))
    y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))
    
    # Initialize a train state, which includes the model paramter and optimizer
    # state.
    model = MLPModel(hidden_dim=dim)
    params = model.init(rngkey, x)
    tx = optax.adam(learning_rate=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    batch = {"x": x, "y": y}

    # print(x.dtype)
    # print(y.dtype)


    expected_state = train_step(state, batch)

    print("[I] Training process on a single device is completed.")

    return state, dim, params, tx, batch, expected_state

#####################################################################
# Step 4. Pipeline Parallelism with Manual Assignment               #
#####################################################################

# Pipeline paralleism requires partitioning the model into several pipeline stages. To manually assign stages, 
# we can use `alpa.mark_pipeline_boundary` to mark the boundary of each pipeline stage in the forward function. 
# Note that each pipeline stage is also automatically parallelized by the shard parallel pass.

# NOTE: Alpa can automatically dispatch training jobs to GPU resources in the Ray cluster. But in this manual mode, 
#       Alpa won't use profile to automatically partition the device cluster (device mesh) to multiple submeshes.
#       - In `./alpa/parallel_method.py/PipeshardParallel(ParallelMethod)/`: 
#       - if layer_option == "manual":
#             layer_option = ManualLayerOption()
#         self.layer_option = layer_option or AutoLayerOption(layer_num=2)
#       - In addition, Alpa supports more flexible manual assignments of pipeline parallelism strategies. 
#         In the above example, each partitioned stages will be assigned an equal number of devices to run. 
#         If you want to control the device assignment of each stage, you can use the more advanced `stage_option=alpa.ManualStageOption`.
#         Then, the `layer_option` is default to `AutoLayerOption(layer_num=2)`.

# NOTE: How to manually assign or impact on the stage-sharding?


def manualPipelineAutoShardMLPTrainOnMultiDev():
    # NOTE: Train on single device to get ground truth
    _, dim, params, tx, batch, expected_state = trainMLPOnSingleDev()

    print("[I] Try MLP train with manually slice pipeline stages & auto-shard on multiple devices.")


    # Define a MLP model with manual stage boundaries.
    class ManualPipelineMLPModel(nn.Module):
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

            # NOTE: THIS IS HOW TO MANUALLY SLICE PIPELINE STAGES!

            # Use this boundary marker to separate the model into two stages.
            alpa.mark_pipeline_boundary()

            x = nn.Dense(features=self.hidden_dim * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
            return x


    # Initialize the train state with the same parameters as the single-device
    # model.
    manual_pipeline_model = ManualPipelineMLPModel(hidden_dim=dim)
    # NOTE: Build from the prev state
    manual_pipeline_state = TrainState.create(apply_fn=manual_pipeline_model.apply,
                                            params=copy.deepcopy(params),
                                            tx=tx)

    # Define the parallel method.
    # We use the "alpa.PipeshardParallel" option to let alpa use both
    # pipeline parallelism and shard parallelism. To make pipeline parallelism
    # efficient, we need to fill the pipeline with many micro batches,
    # so a `num_micro_batches` should be specified for gradient accumulation.
    # `layer_option="manual"` means we use the manually sliced pipeline stages.
    method = alpa.PipeshardParallel(num_micro_batches=16, 
                                    layer_option="manual")
    
    print("[I] Pipeline & Sharding method construction is completed.")

    # Define the training step.
    @alpa.parallelize(method=method)
    def manual_pipeline_train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        # We use `alpa.grad` here to seperate the apply gradient stage with the
        # forward/backward stages in the pipeline. This is necessary to ensure that
        # the gradient accumulation is correct.
        grads = alpa.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state


    manual_pipeline_actual_state = manual_pipeline_train_step(manual_pipeline_state, batch)
    # NOTE: Assert the arrays in x and y are all close.
    assert_allclose(expected_state.params,
                    manual_pipeline_actual_state.params,
                    atol=5e-3)

    print("[I] Training process with manual pipeline stage slicing & auto-sharding on multiple devices is completed. Shutting down alpa...")

    alpa.shutdown()


#####################################################################
# Step 5. Pipeline Parallelism with Automatic Assignment            #
#####################################################################

# Alpa also supports automatically partitioning the model into multiple pipeline stages and assign each pipeline 
# stage a device mesh such that the total execution latency is minimized. Specifically, the automatic partitioning 
# algorithm consists of the following steps:

# 1. Layer Construction: In this step, the operators in the model are clustered into “layers” based on a graph clustering algorithm. 
#    The user needs to specify the total number of layers (i.e. clusters) as a hyperparameter.

# 2. Stage Construction and Mesh Slicing: In this step, we partition the device cluster (device mesh) to multiple submeshes and assign 
#    layers to submeshes to form pipeline stages to minimize the total pipeline execution latency.


def autoPipelineAutoShardMLPTrainOnMultiDev():
    # NOTE: Train on single device to get ground truth
    state, _, _, _, batch, expected_state = trainMLPOnSingleDev()

    print("[I] Try MLP train with auto-slice pipeline stages & auto-shard on multiple devices.")

    # Define the parallel method.
    # `alpa.AutoLayerOption(layer_num=2)` means we use the auto layer construcion
    # algorithm to cluster primitive operators into two layers.
    # `stage_option="auto"` means we enable the auto stage construction algorithm.
    method = alpa.PipeshardParallel(num_micro_batches=16,
                                    layer_option=alpa.AutoLayerOption(layer_num=2),
                                    stage_option="auto")
    
    print("[I] Pipeline & Sharding method construction is completed.")

    # Define the training step. The function body is the same as the above one.
    @alpa.parallelize(method=method)
    def auto_pipeline_train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        # Again, we use `alpa.grad` here to seperate the apply gradient stage with
        # the forward/backward stages in the pipeline.
        grads = alpa.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state


    # In the first call, alpa triggers the compilation.
    # The compilation first profiles several costs and solves an optimization
    # problem to get the optimal pipeline assignments.
    auto_pipeline_actual_state = auto_pipeline_train_step(state, batch)
    assert_allclose(expected_state.params,
                    auto_pipeline_actual_state.params,
                    atol=5e-3)

    print("[I] Training process with automatic pipeline stage slicing & auto-sharding on multiple devices is completed. Shutting down alpa...")

    alpa.shutdown()


if __name__ == "__main__":
    # Option 1: Train MLP on a single device
    # trainMLPOnSingleDev()
    # Option 2: Manual pipeline & auto-sharding on multiple devices
    # manualPipelineAutoShardMLPTrainOnMultiDev()
    # Option 3: Auto pipeline & auto-sharding on multiple devices
    autoPipelineAutoShardMLPTrainOnMultiDev()
