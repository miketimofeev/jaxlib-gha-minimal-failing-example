import jax  # noqa: F401

print(jax)
from jax.config import config

print(config)

config.update("jax_enable_x64", True)
print(config)
import jax.numpy as jnp

print(jnp)

# 32b first
print(jnp.asarray([-2, -1]))
# then switch to 64b
print(jnp.asarray([-2, -1], dtype=jnp.float64))
