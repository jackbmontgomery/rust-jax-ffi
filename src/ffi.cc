#include "xla/ffi/api/ffi.h"
#include "lib.rs.h"
#include "xla/ffi/api/c_api.h"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <type_traits>
#include <utility>

namespace ffi = xla::ffi;
namespace nb = nanobind;

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  size_t size = (size_t)lastDim;
  for (int64_t n = 0; n < totalSize; n += lastDim) {

    const float *x_data = &(x.typed_data()[n]);
    float *y_data = &((*y).typed_data()[n]);

    rust::Slice<const float> x_slice{x_data, size};
    rust::Slice<float> y_slice{y_data, size};

    org::rust_jax_ffi::rms_norm(eps, x_slice, y_slice);
  }
  return ffi::Error::Success();
}

// Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNorm, RmsNormImpl,
                              ffi::Ffi::Bind()
                                  .Attr<float>("eps")
                                  .Arg<ffi::Buffer<ffi::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::F32>>() // y
);

ffi::Error RmsNormFwdImpl(float eps, ffi::Buffer<ffi::F32> x,
                          ffi::ResultBuffer<ffi::F32> y,
                          ffi::ResultBuffer<ffi::F32> res) {

  auto [totalSize, lastDim] = GetDims(x);

  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNormFwd input must be an array");
  }

  size_t size = static_cast<size_t>(lastDim);
  float *res_data = res->typed_data(); // get direct pointer to write scales

  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {

    const float *x_data = &(x.typed_data()[n]);
    float *y_data = &((*y).typed_data()[n]);

    rust::Slice<const float> x_slice{x_data, size};
    rust::Slice<float> y_slice{y_data, size};

    float scale = org::rust_jax_ffi::rms_norm(eps, x_slice, y_slice);
    res_data[idx] = scale;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNormFwd, RmsNormFwdImpl,
                              ffi::Ffi::Bind()
                                  .Attr<float>("eps")
                                  .Arg<ffi::Buffer<ffi::F32>>() // x
                                  .Ret<ffi::Buffer<ffi::F32>>() // y
                                  .Ret<ffi::Buffer<ffi::F32>>() // res
);

ffi::Error RmsNormBwdImpl(ffi::Buffer<ffi::F32> res, ffi::Buffer<ffi::F32> x,
                          ffi::Buffer<ffi::F32> ct_y,
                          ffi::ResultBuffer<ffi::F32> ct_x) {

  auto [totalSize, lastDim] = GetDims(x);

  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNormBwd input must be an array");
  }

  size_t size = static_cast<size_t>(lastDim);
  const float *res_data = res.typed_data();

  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {

    const float *x_data = &(x.typed_data()[n]);
    const float *ct_y_data = &(ct_y.typed_data()[n]);

    float *ct_x_data = &((*ct_x).typed_data()[n]);
    float scale = res_data[idx];

    rust::Slice<const float> x_slice{x_data, size};
    rust::Slice<const float> ct_y_slice{ct_y_data, size};
    rust::Slice<float> ct_x_slice{ct_x_data, size};

    org::rust_jax_ffi::rms_norm_bwd(scale, x_slice, ct_y_slice, ct_x_slice);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNormBwd, RmsNormBwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>() // res
                                  .Arg<ffi::Buffer<ffi::F32>>() // x
                                  .Arg<ffi::Buffer<ffi::F32>>() // ct_y
                                  .Ret<ffi::Buffer<ffi::F32>>() // ct_x
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(rust_jax_ffi, m) {
  m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
  m.def("rms_norm_fwd", []() { return EncapsulateFfiCall(RmsNormFwd); });
  m.def("rms_norm_bwd", []() { return EncapsulateFfiCall(RmsNormBwd); });
}
