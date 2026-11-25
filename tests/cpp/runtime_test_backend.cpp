// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {
struct Context { int device; };
struct Kernel { int device; std::string symbol; };

void trace(const char* action, int device, const std::string& payload = "") {
  std::ofstream out(std::getenv("FLAGCV_TEST_BACKEND_TRACE"), std::ios::app);
  out << action << " " << device << payload << '\n';
}

int create(int device, void** context) {
  *context = new Context{device};
  trace("create", device);
  return 0;
}

void destroy(void* raw) {
  auto* context = static_cast<Context*>(raw);
  trace("destroy", context->device);
  delete context;
}

int device_id(void* raw, char* output, std::size_t size) {
  std::snprintf(output, size, "fixture-device-%d", static_cast<Context*>(raw)->device);
  return 0;
}

int target(void*, char* output, std::size_t size) {
  std::snprintf(output, size, "cuda:90:32");
  return 0;
}

int load(void* raw, const void*, std::size_t size, const char* symbol,
         unsigned int, unsigned int, void** out) {
  const auto device = static_cast<Context*>(raw)->device;
  if (size == 0) return 1;
  *out = new Kernel{device, symbol};
  trace("load", device, " " + std::string(symbol));
  return 0;
}

void unload(void* raw, void* handle) {
  auto* kernel = static_cast<Kernel*>(handle);
  trace("unload", static_cast<Context*>(raw)->device, " " + kernel->symbol);
  delete kernel;
}

int launch(void* raw, void* handle, const unsigned int grid[3],
           unsigned int, unsigned int, void* stream, void** args, const unsigned int* kinds, unsigned int count) {
  const auto device = static_cast<Context*>(raw)->device;
  auto* kernel = static_cast<Kernel*>(handle);
  if (kernel->device != device) return 1;
  if (const char* fail = std::getenv("FLAGCV_TEST_FAIL_KERNEL")) {
    if (kernel->symbol == fail) {
      trace("rejected", device, " " + kernel->symbol);
      return 1;
    }
  }
  if (kernel->symbol == "threshold_select") {
    std::uint64_t address;
    std::memcpy(&address, args[0], sizeof(address));
    *reinterpret_cast<std::int32_t*>(address) = -1;
  }

  if (kernel->symbol == "_warp_nearest") {
    if (count != 18 || kinds[0] != 0 || kinds[1] != 0) return 1;
    for (unsigned int i = 2; i < count; ++i)
      if (kinds[i] != 1) return 1;
    std::ostringstream values;
    for (int i : {2, 3, 4, 5, 6, 7, 14, 15, 16, 17})
      values << ' ' << *static_cast<float*>(args[i]);
    trace("fp32", device, values.str());
  }
  trace("launch", device, " " + kernel->symbol + " " + std::to_string(grid[0]) +
                              (stream ? " custom" : " default"));
  return 0;
}

const char* last_error() { return "runtime test backend rejected invalid launch"; }
}  // namespace

extern "C" const FlagcvBackendV2* flagcv_backend_v2() {
  static const FlagcvBackendV2 backend{FLAGCV_BACKEND_VERSION, 0, create, destroy, device_id,
                                      target, load, unload, launch, last_error};
  return &backend;
}

#ifdef FLAGCV_TEST_READBACK
extern "C" int flagcv_readback_v1(void* context, const void* source, void* host,
                                 size_t size, void* stream) {
  trace("readback", static_cast<Context*>(context)->device, stream ? " custom" : " default");
  std::memcpy(host, source, size);
  return 0;
}
#endif
