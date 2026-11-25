// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/backend.hpp"

#include <cstdio>

namespace {
int create(int, void** context) {
  static int fixture;
  *context = &fixture;
  return 0;
}
void destroy(void*) {}
int device_id(void*, char* buffer, size_t size) {
  std::snprintf(buffer, size, "scratch-fixture");
  return 0;
}
int target(void*, char* buffer, size_t size) {
  std::snprintf(buffer, size, "cuda:90:32");
  return 0;
}
int load(void*, const void*, size_t, const char*, unsigned, unsigned, void**) {
  // The sentinel error proves which requests reached the device boundary.
  return 1;
}
const char* last_error() { return "scratch fixture reached module load"; }
}  // namespace

extern "C" const FlagcvBackendV2* flagcv_backend_v2() {
  static const FlagcvBackendV2 fixture{FLAGCV_BACKEND_VERSION, 1, create, destroy,
                                       device_id, target, load, nullptr, nullptr,
                                       last_error};
  return &fixture;
}
