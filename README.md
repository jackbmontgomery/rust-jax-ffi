# Rust Jax FFI


## Issues
1. When running normally with:
```
cmake -S . -B build
cmake --build build
```
It gives the error:
```
clang++: error: no such file or directory: '/Users/jackmontgomery/personal/rust-jax-ffi/target/cxxbridge/rust_jax_ffi/src/lib.rs.cc'
clang++: error: no input files
```

It can be resolved by saving the build.rs file and rerunning `cmake --build build`.

2. I want to be able to automatically pickup the resources built for python, now I need to copy them into the root to test them.

I think I should add python to the PYPATH during the build?
