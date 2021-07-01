print("Entered into debug_32.py")
import jax  # noqa: F401

print("Ran: import jax")

print(jax)
print("Ran: print(jax)")
import jax.numpy as jnp

print("Ran: import jax.numpy as jnp")

print(jnp)
print("Ran: print(jnp)")

print(jnp.asarray([-2, -1], dtype=jnp.float32))
print("Ran: print(jnp.asarray([-2, -1], dtype=jnp.float32))")
print(jnp.asarray([-2, -1], dtype=jnp.float64))
print("Ran: print(jnp.asarray([-2, -1], dtype=jnp.float64))")
