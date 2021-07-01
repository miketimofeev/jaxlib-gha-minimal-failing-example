import jax  # noqa: F401

print(jax)
import jax.numpy as jnp

print(jnp)

print(jnp.asarray([-2, -1], dtype=jnp.float32))
print(jnp.asarray([-2, -1], dtype=jnp.float64))
