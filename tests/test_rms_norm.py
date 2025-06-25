import jax
import jax.numpy as jnp
import numpy as np


def test_rms_norm(rms_norm, rms_norm_ref):
    x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
    y = rms_norm(x)
    y_ref = rms_norm_ref(x)
    np.testing.assert_allclose(y, y_ref, rtol=1e-5)


def test_rms_norm_vmap(rms_norm, rms_norm_ref):
    x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
    np.testing.assert_allclose(
        jax.vmap(rms_norm)(x), jax.vmap(rms_norm_ref)(x), rtol=1e-5
    )


def test_rms_norm_(rms_norm, rms_norm_fwd, rms_norm_bwd, rms_norm_ref):
    rms_norm = jax.custom_vjp(rms_norm, nondiff_argnums=(1,))
    rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)

    x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
    ct_y = jnp.ones_like(x)

    _, vjp_custom = jax.vjp(rms_norm, x)
    _, vjp_ref = jax.vjp(rms_norm_ref, x)

    grad_custom = vjp_custom(ct_y)[0]
    grad_ref = vjp_ref(ct_y)[0]

    print("grad_custom (from FFI):\n", grad_custom)
    print("grad_ref (from reference):\n", grad_ref)

    np.testing.assert_allclose(grad_custom, grad_ref, rtol=1e-5)
