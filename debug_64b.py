import jax  # noqa: F401
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

# 32b first
jnp.asarray([-2, -1])
# then switch to 64b
jnp.asarray([-2, -1], dtype=jnp.float64)
