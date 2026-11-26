// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/backend.hpp"
#include "src/runtime/paths.hpp"

#include <dlfcn.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace live {
struct Device {
  std::string target;
  std::string key;
  std::string plugin;
  int abi;
  int scratch_arguments;
};

inline Device plugin_device() {
  const auto path = flagcv::plugin_path(FLAGCV_LIVE_BACKEND);
  const auto close_library = [](void* handle) { (void)dlclose(handle); };
  std::unique_ptr<void, decltype(close_library)> library(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), close_library);
  if (!library) throw std::runtime_error(dlerror());
  const auto entry = reinterpret_cast<FlagcvBackendEntry>(dlsym(library.get(), FLAGCV_BACKEND_ENTRY));
  const auto* api = entry ? entry() : nullptr;
  if (!api || api->version != FLAGCV_BACKEND_VERSION)
    throw std::runtime_error("incorrect live backend ABI");
  void* ctx = nullptr;
  if (api->create_context(0, &ctx) != 0) throw std::runtime_error(api->last_error());
  const std::unique_ptr<void, decltype(api->destroy_context)> context(ctx, api->destroy_context);
  char target[256], key[256];
  if (api->target(ctx, target, sizeof(target)) || api->device_id(ctx, key, sizeof(key)))
    throw std::runtime_error(api->last_error());
  return {target, key, path, api->version, api->appends_scratch_args ? 2 : 0};
}

inline const Device& device() {
  static const Device result = plugin_device();
  return result;
}
}  // namespace live
