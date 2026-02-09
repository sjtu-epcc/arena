# Slurm Tutorials

###### References

 - [上海交大交我算HPC+AI平台用户文档](https://docs.hpc.sjtu.edu.cn/index.html)
 - [Pi2.0 超算平台用户手册](https://docs.hpc.sjtu.edu.cn/job/slurm.html)
 - [Slurm 作业调度系统使用指南](https://zhuanlan.zhihu.com/p/356415669)

-----------

## 1. Batch job

 - Command: 
    - `sbatch job.slurm`
        Suitable for batch jobs, not recommended for jobs that need pre-building to construct environment (since not easy to debug).
 - NOTE: 
    - The execution path of .slurm script is in `/tmp` and not in the orginal path where the script is located.

 ----------

 ## 2. Interaction job

 - Command: 
    - `srun -p [QUEUE_NAME] -n [CORE_NUM] -N [NODE_NUM] --pty /bin/bash`
        Automatically connect to the assigned node and release resources after exiting from the node.
    - `salloc -p [QUEUE_NAME] -n [CORE_NUM] -N [NODE_NUM]`
        Given node alias, use ssh command to enter or exit the node, need to manually release the assigned resources.

