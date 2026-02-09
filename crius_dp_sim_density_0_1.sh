python simulator.py --policy=fcfs --trace_type=philly --enable_alpa --max_sched_round=2000 --job_submit_density=0.1 --result_dir=./plot_revision_0_1
python simulator.py --policy=gavel --trace_type=philly --enable_alpa --max_sched_round=2000 --job_submit_density=0.1 --result_dir=./plot_revision_0_1
python simulator.py --policy=elasticflow-l --trace_type=philly --enable_alpa --max_sched_round=2000 --job_submit_density=0.1 --result_dir=./plot_revision_0_1
python simulator.py --policy=sia --trace_type=philly --enable_alpa --max_sched_round=2000 --job_submit_density=0.1 --result_dir=./plot_revision_0_1
python simulator.py --policy=crius-dp --trace_type=philly --enable_alpa --max_sched_round=2000 --job_submit_density=0.1 --result_dir=./plot_revision_0_1
