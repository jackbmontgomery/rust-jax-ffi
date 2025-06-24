import jax
import jax.numpy as jnp
import numpy as np
import rust_jax_ffi

jax.ffi.register_ffi_target("rms_norm", rust_jax_ffi.rms_norm(), platform="cpu")


def rms_norm_ref(x, eps=1e-5):
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / scale


def rms_norm(x, eps=1e-5):
    # We only implemented the `float32` version of this function, so we start by
    # checking the dtype. This check isn't strictly necessary because type
    # checking is also performed by the FFI when decoding input and output
    # buffers, but it can be useful to check types in Python to raise more
    # informative errors.
    if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")

    call = jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "rms_norm",
        # In this case, the output of our FFI function is just a single array with
        # the same shape and dtype as the input. We discuss a case with a more
        # interesting output type below.
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        # The `vmap_method` parameter controls this function's behavior under `vmap`
        # as discussed below.
        vmap_method="broadcast_all",
    )

    # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
    # the attribute `eps`. Our FFI function expects this to have the C++ `float`
    # type (which corresponds to numpy's `float32` type), and it must be a
    # static parameter (i.e. not a JAX array).
    return call(x, eps=np.float32(eps))


# Test that this gives the same result as our reference implementation
x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
np.testing.assert_allclose(rms_norm(x), rms_norm_ref(x), rtol=1e-5)

print("✅ Test passed! RMS norm FFI implementation is correct.")
