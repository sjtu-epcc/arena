ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=10.2.64.81:6379 --num-gpus 1 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
