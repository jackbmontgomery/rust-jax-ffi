import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import rust_jax_ffi


@pytest.fixture(scope="session", autouse=True)
def register_rust_ffi():
    jax.ffi.register_ffi_target("rms_norm", rust_jax_ffi.rms_norm(), platform="cpu")
    jax.ffi.register_ffi_target(
        "rms_norm_fwd", rust_jax_ffi.rms_norm_fwd(), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "rms_norm_bwd", rust_jax_ffi.rms_norm_bwd(), platform="cpu"
    )


@pytest.fixture
def rms_norm_ref():
    def fn(x, eps=1e-5):
        scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return x / scale

    return fn


@pytest.fixture
def rms_norm():
    def fn(x, eps=1e-5):
        if x.dtype != jnp.float32:
            raise ValueError("Only the float32 dtype is implemented by rms_norm")

        call = jax.ffi.ffi_call(
            "rms_norm",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )

        return call(x, eps=np.float32(eps))

    return fn


@pytest.fixture
def rms_norm_fwd():
    def fn(x, eps=1e-5):
        y, res = jax.ffi.ffi_call(
            "rms_norm_fwd",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),
            ),
            vmap_method="broadcast_all",
        )(x, eps=np.float32(eps))
        return y, (res, x)

    return fn


@pytest.fixture
def rms_norm_bwd():
    def fn(eps, res, ct):
        del eps
        res, x = res
        assert res.shape == ct.shape[:-1]
        assert x.shape == ct.shape
        return (
            jax.ffi.ffi_call(
                "rms_norm_bwd",
                jax.ShapeDtypeStruct(ct.shape, ct.dtype),
                vmap_method="broadcast_all",
            )(res, x, ct),
        )

    return fn
