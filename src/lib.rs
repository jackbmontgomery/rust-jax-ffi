fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) -> f32 {
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

    scale
}

fn rms_norm_bwd(res: f32, x: &[f32], ct_y: &[f32], ct_x: &mut [f32]) {
    debug_assert_eq!(x.len(), ct_y.len(), "x and ct_y must have the same length");
    debug_assert_eq!(x.len(), ct_x.len(), "x and ct_x must have the same length");

    let size = x.len();
    let mut dot = 0f32;

    for i in 0..size {
        dot += x[i] * ct_y[i];
    }

    let factor = dot * res * res * res / size as f32;

    for i in 0..size {
        ct_x[i] = res * ct_y[i] - factor * x[i];
    }
}

#[cxx::bridge(namespace = "org::rust_jax_ffi")]
mod ffi {
    extern "Rust" {
        fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) -> f32;
        fn rms_norm_bwd(res: f32, x: &[f32], ct_y: &[f32], ct_x: &mut [f32]) -> ();
    }
}
