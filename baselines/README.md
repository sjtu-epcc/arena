# Baseline Descriptions

Currently, we offer the following baselines:
 - **FCFS**: Basic first-come-first-serve policy that does not modify GPU type or amount specified by each job. 
 - **Gandiva**: Gandiva utilizes domain-specific knowledge to refine scheduling decisions based on runtime profiling introspectively. Like FCFS, initially it does not modify GPU type or amount, only online reschedule them. While it could adjust the GPU quota and topology, it ignores the GPU heterogeneity. 
 - **Gavel**: Gavel is a heterogeneity-aware scheduler designed for various scheduling policies, including throughput maximization. Although it considers the GPU heterogeneity, it does not support the scaling of GPU amount. 
 - **ElasticFlow**: ElasticFlow is an adaptivity-aware scheduler for elastic job scaling in a homogeneous cluster, with a deadline-aware policy and a throughput-oriented policy.
 - **Sia**: TODO

> It should be noted that, since FCFS and Gandiva would not modify GPU type and amount of each newly submitted job, **we directly use the performance of optimal parallelism when determining whether each job can be allocated**. For ElasticFlow, Gavel and Sia, since they modify GPU amount or type initially, we **use the performance of data parallelism during their scheduling**, and **use performance of optimal parallelism when determining whether these scheduled jobs can be really deployed** (e.g., without OOM, without uneven locality).
