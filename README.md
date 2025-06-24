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
