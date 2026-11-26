/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <hip/hip_runtime.h>

#include "backends/common/driver_backend.hpp"

// Internal HIP driver shared by the separate ROCm and DTK plugins.
#include <string_view>

namespace {

constexpr bool supported_target(std::string_view vendor, std::string_view arch, int wave) {
  if (vendor == "amd") return arch == "gfx1100" && wave == 32;
  return vendor == "hygon" && (arch == "gfx936" || arch == "gfx938") && wave == 64;
}

static_assert(supported_target("amd", "gfx1100", 32) &&
              supported_target("hygon", "gfx936", 64) &&
              supported_target("hygon", "gfx938", 64));
static_assert(!supported_target("amd", "gfx936", 64) &&
              !supported_target("amd", "gfx938", 64) &&
              !supported_target("hygon", "gfx1100", 32) &&
              !supported_target("hygon", "gfx936", 32) &&
              !supported_target("amd", "gfx942", 64) &&
              !supported_target("hip", "gfx936", 64));

struct HipApi {
  using Status = hipError_t;
  using Device = hipDevice_t;
  using Context = int;
  using Module = hipModule_t;
  using Function = hipFunction_t;

  static const char* name() { return FLAGCV_HIP_VENDOR; }
  static Status success() { return hipSuccess; }

  static Status init() { return hipInit(0); }
  static Status device_get(Device* device, int index) {
    return hipDeviceGet(device, index);
  }
  static Status primary_ctx_retain(Context* ctx, Device device) {
    *ctx = device;
    return hipSuccess;
  }
  static Status primary_ctx_release(Device) { return hipSuccess; }
  static Status module_load_data(Module* module, const void* data) {
    return hipModuleLoadData(module, data);
  }
  static Status module_get_function(Function* fn, Module module,
                                    const char* name) {
    return hipModuleGetFunction(fn, module, name);
  }
  static Status module_unload(Module module) { return hipModuleUnload(module); }
  static Status launch_kernel(Function fn, unsigned gx, unsigned gy,
                              unsigned gz, unsigned block, unsigned shared,
                              void* stream, void** args) {
    // hipModuleLaunchKernel takes the stream by type and the extra
    // shared-memory/stream arguments Triton does not use.
    return hipModuleLaunchKernel(fn, gx, gy, gz, block, 1, 1, shared,
                                 static_cast<hipStream_t>(stream), args,
                                 nullptr);
  }
  static Status set_shared(Function fn, int bytes) {
    return hipFuncSetAttribute(fn, hipFuncAttributeMaxDynamicSharedMemorySize,
                               bytes);
  }

  static Status attribute(int* value, hipDeviceAttribute_t what,
                          Device device) {
    return hipDeviceGetAttribute(value, what, device);
  }

  static Status target(Device device, std::string& out) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return hipErrorUnknown;
    // gcnArchName carries target features ("gfx936:sramecc+:xnack-"); Triton
    // takes the bare architecture, as its own HIP driver does.
    std::string arch = props.gcnArchName;
    arch = arch.substr(0, arch.find(':'));
    if (!supported_target(name(), arch, props.warpSize)) return hipErrorInvalidDevice;
    out = "hip:" + arch + ":" + std::to_string(props.warpSize);
    return hipSuccess;
  }

  static Status device_key(Device device, std::string& out) {
    hipDeviceProp_t props{};
    int bus = 0;
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return hipErrorUnknown;
    if (hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device) != hipSuccess)
      return hipErrorUnknown;
    out =
        std::string(name()) + ":" + props.gcnArchName + ":bus" + std::to_string(bus);
    return hipSuccess;
  }

  static int shared_optin_limit(Device device) {
    // Use the portable limit common to ROCm and DTK. The currently shipped
    // image kernels do not need a larger vendor-specific opt-in allocation.
    int limit = 0;
    if (attribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
      return 0;
    return limit;
  }
  static unsigned shared_optin_threshold() { return 48u * 1024u; }
};

}  // namespace

namespace flagcv::driver {

// HIP's supported device APIs are thread-local. Bind each driver operation
// to the Handle's device and restore the caller's device without resetting it.
template <>
class Scope<HipApi> {
 public:
  explicit Scope(int device) {
    if (hipGetDevice(&previous_) != hipSuccess) return;
    if (previous_ == device) {
      bound_ = true;
    } else {
      restore_ = hipSetDevice(device) == hipSuccess;
      bound_ = restore_;
    }
  }
  ~Scope() {
    if (restore_) (void)hipSetDevice(previous_);
  }
  bool ok() const { return bound_; }
 private:
  int previous_ = 0;
  bool bound_ = false;
  bool restore_ = false;
};

}  // namespace flagcv::driver

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  return flagcv::driver::table<HipApi>();
}
