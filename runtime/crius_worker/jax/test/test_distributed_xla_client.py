"""
Ref: https://gist.github.com/EiffL/e0fbc18007bf139dd66124f971144821
"""

# On the clients
from jax.lib import xla_client as xc
client = xc._xla.get_distributed_runtime_client("dns:///localhost:8484")
back_end = xc._gpu_backend_factory(client, node_id=0) # or node_id=1
# This returns a backend that can be used with Jax apparently

# For instance, to use the returned backend for jax by default on a client
import jax
jax.lib.xla_bridge.register_backend('mygpu', lambda x: back_end)
jax.config.update('jax_xla_backend', 'mygpu')