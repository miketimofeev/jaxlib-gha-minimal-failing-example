print("Entered into debug_64.py")
import jax  # noqa: F401

print("Ran: import jax")

print(jax)
print("Ran: print(jax)")
from jax.config import config

print("Ran: from jax.config import config")

print(config)
print("Ran: print(config)")

config.update("jax_enable_x64", True)
print("Ran: config.update('jax_enable_x64', True)")
print(config)
print("Ran: print(config)")
import jax.numpy as jnp

print("Ran: import jax.numpy as jnp")

print(jnp)
print("Ran: print(jnp)")

# 32b first
print(jnp.asarray([-2, -1]))
print("Ran: print(jnp.asarray([-2, -1]))")
# then switch to 64b
print(jnp.asarray([-2, -1], dtype=jnp.float64))
print("Ran: print(jnp.asarray([-2, -1], dtype=jnp.float64))")
