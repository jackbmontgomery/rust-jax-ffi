#include "xla/ffi/api/ffi.h"
#include "lib.rs.h"
#include "xla/ffi/api/c_api.h"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <type_traits>
#include <utility>

namespace ffi = xla::ffi;

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

namespace nb = nanobind;

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(rust_jax_ffi, m) {
  m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
}
