fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) {
    println!("Hello From Rust");
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let mut sm = 0f32;
    let size = x.len();
    for xi in x {
        sm += xi * xi;
    }
    let scale = (sm / (size as f32) + eps).sqrt().recip();

    for i in 0..size {
        y[i] = x[i] * scale;
    }
}

#[cxx::bridge(namespace = "org::rust_jax_ffi")]
mod ffi {
    extern "Rust" {
        fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) -> ();
    }
}
