# Rust Jax FFI

## Setup Instructions

1. Clone or fork the repo then run:
```
git clone https://github.com/<GITHUB_USERNAME>/rust-jax-ffi.git
```
2. Change directory into it
3. `uv venv`
4. `uv sync`
5. `source .venv/bin/activate`
6. `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
    - This will take a while
7. `cmake --build build`

## Issues

1. I want to be able to automatically pickup the resources built for python, now I need to copy them into the root to test them.
    - I think I should add python to the PYPATH during the build?

## TODO

- Implement some pytesting
- export the package into a nice place in the build process


## Limitations

At this point, we can use our new rms_norm function transparently for many JAX applications, and it will transform appropriately under the standard JAX function transformations like vmap() and grad(). One thing that this example doesn’t support is forward-mode AD (jax.jvp(), for example) since custom_vjp() is restricted to reverse-mode. JAX doesn’t currently expose a public API for simultaneously customizing both forward-mode and reverse-mode AD, but such an API is on the roadmap, so please open an issue describing you use case if you hit this limitation in practice.

- https://docs.jax.dev/en/latest/ffi.html#differentiation

