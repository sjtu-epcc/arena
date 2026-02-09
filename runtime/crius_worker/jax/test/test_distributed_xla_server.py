"""
Server end
------------------
- Ref: https://gist.github.com/EiffL/e0fbc18007bf139dd66124f971144821
- This script is running on 10.2.64.51 to start RPC server on 10.2.64.52.
"""


import random
from jax.lib import xla_client

max_trials = 10
cnt = 0

while cnt < max_trials:
    cnt += 1

    # Random port
    port = random.randint(20000, 25000)
    # Address
    addr = '10.2.64.52:' + str(port)

    print("")
    print("[I] The target server address is:", addr)
    print("")

    try: 
        server = xla_client._xla.get_distributed_runtime_service(address=addr, 
                                                    num_nodes=2,
                                                    use_coordination_service=False)
        print("[I] Successfully start RPC server on {}".format(addr))
        exit(0)
    except:
        print("[I] Trial {}/{}: Failed to start RPC server on {}".format(cnt, max_trials, addr))
