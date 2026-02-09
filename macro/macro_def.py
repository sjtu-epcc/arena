#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script to record all definition of macros universally used in the project.
"""

########################################
#        Global Configurations         #
########################################
# Precision
PREC = 3
# Universal is_succeed flag
IS_SUCCEED = True
# Universal is_failed flag
IS_FAILED = False
# Universal hardware identity
IS_NODE = 'node'
IS_GPU = 'gpu'


########################################
#         Hardware Abstraction         #
########################################
# Init value
INIT_USED_MEM, INIT_USED_BW, INIT_GPU_UTIL = 0.0, 0.0, 0.0
# Max value
MAX_GPU_UTIL = 1.0

# GPU status table
GPU_STATUS_TABLE = ( 'USED', 'RESERVED', 'IDLE' )
USED_STATUS = 'USED'
# RESERVED_STATUS = 'RESERVED'
IDLE_STATUS = 'IDLE'


########################################
#          Job Encapsulation           #
########################################
# Job status table
JOB_STATUS_TABLE = ( 'SUBMITTED', 'INIT_READY', 'PENDING', 'RUNNING', 'SUSPENDED', 'COMPLETED', 'ERROR' )
JOB_SUBMITTED_STATUS = 'SUBMITTED'
JOB_INIT_READY_STATUS = 'INIT_READY'
JOB_PENDING_STATUS = 'PENDING'
JOB_RUNNING_STATUS = 'RUNNING'
# JOB_SUSPENDED_STATUS = 'SUSPENDED'
JOB_COMPLETED_STATUS = 'COMPLETED'
JOB_ERROR_STATUS = 'ERROR'
# Empty job ID
EMPTY_JOB_ID = ''
# Empty job alias
EMPTY_JOB_ALIAS = 'empty'
# Fake job ID
FAKE_JOB_ID = 'fake_job_id'
# Init job num
INIT_JOB_NUM = 0


########################################
#         Parallelism Related          #
########################################
# Candidate parallelism flags
CAND_PARALLEL_FLAGS = (
    "opt", "pp", "dp", "mp"
)


########################################
#           Profiler Related           #
########################################
# Scaling factor for varying locality
LOCALITY_SCALE_FACTOR = 0.9
# Linear scaling factor for varying param num
PARAM_NUM_SCALE_FACTOR = 1.8
# Log coefficient used in predicting the scaling curve of the throughput
LOG_COE = 1.0
# Maximum GPU num for each GPU type
MAX_GPU_NUM_TABLE = {
    "v100": 16,
    "a100": 8,
}


########################################
#          Dummy Test Related          #
########################################
# Max shrink times
MAX_SHRINK_TIMES = 2


########################################
#          Scheduler Related           #
########################################
# TODO(chunyu): Re-formulate modeling of (1) profiling overhead; (2) ckpt-resume overhead; (3) auto-parallel search overhead.

# Infeasible structured marginal gain
INFEASIBLE_THR = -1e9
# Iteration num of job shrink search (2 for simulation while 3 for runtime)
ITER_NUM_OF_DOWNGRADE_SEARCH = 2
# Modified flags in job shrink and hardware type change search
IS_NO_MODIFIED = 'no_modified'
IS_SHRINKED = 'shrinked'
IS_HTC = 'hardware_type_changed'
# Max supported GPU num
MAX_SUPPORTED_GPU_NUM = 16
SUPPORTED_GPU_NUM_LIST = [1, 2, 4, 8, 16]
# Is job arrival or departure event
IS_ARRIVAL_EVENT = 'arrival_event'
IS_DEPARTURE_EVENT = 'departure_event'
# Scheduling interval for simulation (in seconds)
SCHEDULING_INTERVAL = 300
# Infeasible iteration time
INFEASIBLE_ITER_TIME = 1e9
# Maximal upgrading step num
MAX_UPGRADE_STEP_NUM = 10
# Maximal restart trial times
MAX_RESTART_TRIAL_NUM = 100
# Per-gpu resched time overhead (for Alpa search)
RESCHED_OVERHEAD_WITH_PRUNE = 10
MAX_RESCHED_OVERHEAD_WITH_PRUNE = 40
RESCHED_OVERHEAD_WITHOUT_PRUNE = 150
RESCHED_OVERHEAD_WITHOUT_PRUNE_RT = 150
MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE = 600
MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE_RT = 400
# Per-gpu ckpt-resume time overhead
# CKPT_RESUME_OVERHEAD = 50
CKPT_RESUME_OVERHEAD = 0
MAX_CKPT_RESUME_OVERHEAD = 400
# Per-config profile overhead
DP_PROFILE_OVERHEAD_PER_CONFIG = 30
AP_PROFILE_OVERHEAD_PER_CONFIG = 60


########################################
#           Baseline Related           #
########################################
# The extreme-long deadline for the jobs in ElasticFlow baseline
LOOSEN_DEADLINE = 1e9
# The infeasible max throughput for jobs in Gavel baseline 
MAX_THR = 1e9
# Deadline loosen ratio
DDL_LOOSEN_RATIO = 1.5


########################################
#           Runtime Related            #
########################################
# Migration configurations, including the hyperparameters that restrict the migration upper bound.
LOCAL_ETA = 256
GLOBAL_ETA = 4096
# Migration configurations, including the hyperparameters that restrict the migration upper bound.
MGRT_CFGS = {
    'local_eta': LOCAL_ETA,
    'global_eta': GLOBAL_ETA,
}
# Max retry times to send request to the node
MAX_RETRY_TIMES = 20
# Runtime operation type
IS_DISPATCH = 'is_dispatch'
IS_SUSPEND = 'is_suspend'
IS_CHECK = 'is_check'
# Scheduling interval
RUNTIME_SCHEDULING_INTERVAL = 300
# Check interval
RUNTIME_CHECK_INTERVAL = 20
# Runtime max iter num
RUNTIME_MAX_ITER_NUM = 500
# Memory utilzation threshold for forcibly apply preferred parallel method
PPM_MEM_UTIL_THRESHOLD = 0.3
# Memory utilzation threshold for providing searching prompts
PROMPT_MEM_UTIL_THRESHOLD = 0.6
# Otherwise, recommanding stage num = len(locality)
# Prompt type
IS_FORCE_DP = 'force_dp'
IS_FORCE_PP = 'force_pp'
IS_REC_STAGE_NUM = 'rec_stage_num'
# End event type
IS_END = 'end'
IS_ERROR = 'error'


########################################
#            Trace Related             #
########################################
# Scale factor for iter num in runtime
ITER_NUM_SCALE_FACTOR = 100
# Scale factor for iter num in simulation
ITER_NUM_SCALE_FACTOR_SIM = 5
# Min and max iter num for helios venus trace
MIN_ITER_NUM_VENUS, MAX_ITER_NUM_VENUS = 2000, 10000
# Min and max iter num for pai trace
MIN_ITER_NUM_PAI, MAX_ITER_NUM_PAI = 1000, 5000
