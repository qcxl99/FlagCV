/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/runtime.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory_resource>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include "backends/backend.hpp"
#include "src/runtime/compiler_client.hpp"
#include "src/runtime/json.hpp"
#include "src/runtime/paths.hpp"

namespace flagcv {

namespace {

const char* depth_name(Depth depth) {
  switch (depth) {
    case Depth::k8U:
      return "8U";
    case Depth::k8S:
      return "8S";
    case Depth::k16U:
      return "16U";
    case Depth::k16S:
      return "16S";
    case Depth::k32S:
      return "32S";
    case Depth::k16F:
      return "16F";
    case Depth::k32F:
      return "32F";
    case Depth::k64F:
      return "64F";
  }
  throw Error("flagcv: unknown pixel depth");
}

/*
 * A loaded plugin. Plugins are opened once per process and never closed:
 * unloading a vendor runtime while its threads or atexit handlers are live is
 * a common source of shutdown crashes, and the mapping is small.
 */
const FlagcvBackendV2& load_plugin(const std::string& backend) {
  static std::mutex mutex;
  static std::map<std::string, const FlagcvBackendV2*> plugins;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = plugins.find(backend);
  if (it != plugins.end()) return *it->second;

  if (backend.empty() || backend.find_first_not_of('_') == std::string::npos)
    throw Error("flagcv: invalid backend name '" + backend + "'");
  for (char c : backend) {
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
      throw Error("flagcv: invalid backend name '" + backend + "'");
    }
  }
  const std::string path = plugin_path(backend);
  void* library = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    throw Error("flagcv: cannot load backend plugin " + path + ": " +
                ::dlerror() + " (set FLAGCV_PLUGIN_DIR to override)");
  }
  auto entry = reinterpret_cast<FlagcvBackendEntry>(
      ::dlsym(library, FLAGCV_BACKEND_ENTRY));
  if (entry == nullptr) {
    ::dlclose(library);
    throw Error("flagcv: " + path + " does not export " FLAGCV_BACKEND_ENTRY);
  }
  const FlagcvBackendV2* table = entry();
  if (table == nullptr || table->version != FLAGCV_BACKEND_VERSION) {
    ::dlclose(library);
    throw Error("flagcv: " + path +
                " was built against a different plugin interface version");
  }
  plugins.emplace(backend, table);
  return *table;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("flagcv: cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/* One grid dimension: a fixed extent, or ceil(product of params / block). */
struct GridDim {
  std::int64_t fixed = 1;
  std::vector<std::size_t> product;
  std::int64_t block = 1;
  std::int64_t limit = std::numeric_limits<std::int64_t>::max();
};

struct LoadedKernel {
  std::string_view name;
  std::vector<std::int64_t> signature;
  void* kernel = nullptr;
  unsigned int block_threads = 0;
  unsigned int shared_bytes = 0;
  GridDim grid[3];
};

/* Compiler identity per backend, asked once per process. */
const std::string& compiler_identity(const std::string& backend) {
  static std::mutex mutex;
  static std::map<std::string, std::string> identities;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = identities.find(backend);
  if (it == identities.end()) {
    json::Value reply = CompilerClient::instance().request(
        "\"op\":\"identify\",\"backend\":" + json::quote(backend));
    it = identities.emplace(backend, reply.at("identity").as_string()).first;
  }
  return it->second;
}

bool aligned16(std::uint64_t address) { return address % 16 == 0; }

bool fits_i32(std::int64_t value) {
  return value >= std::numeric_limits<std::int32_t>::min() &&
         value <= std::numeric_limits<std::int32_t>::max();
}

/* Runtime params are specialised on width, and on divisibility by 16 when
 * the lowering asked for it (LoweredOp::Param::divisibility_hint). */
const char* runtime_type(std::int64_t value) {
  return fits_i32(value) ? "i32" : "i64";
}
bool divisible16(const LoweredOp::Param& param) {
  return param.divisibility_hint && param.value % 16 == 0;
}

std::int64_t signature_value(const ImageSpec& image) {
  return (std::int64_t{image.channels} << 8) |
         (static_cast<int>(image.depth) << 1) | aligned16(image.address);
}

std::int64_t signature_value(const LoweredOp::Param& param) {
  // For each registry name, the ordering and constexpr positions are fixed
  // by lowering. Runtime values contribute their ABI and alignment only.
  return param.compile_time ? param.value
                            : (fits_i32(param.value) << 1) | divisible16(param);
}

}  // namespace

struct Handle::State {
  std::string backend;
  int device = 0;
  const FlagcvBackendV2* plugin = nullptr;
  void* context = nullptr;
  std::string device_id;
  std::string target;
  std::string compiler_identity;  // fetched on the first launch
  std::unordered_multimap<std::size_t, LoadedKernel> kernels;

  ~State() {
    // The cache goes before the plugin context it was loaded into.
    if (plugin == nullptr) return;
    for (auto& entry : kernels) plugin->unload(context, entry.second.kernel);
    kernels.clear();
    if (context != nullptr) plugin->destroy_context(context);
  }
};

struct HandleAccess {
  static Handle::State& state(Handle& handle) {
    if (!handle.state_) throw Error("flagcv: handle was moved from");
    return *handle.state_;
  }
};

Handle::Handle(const std::string& backend, int device) : state_(new State) {
  state_->backend = backend;
  state_->device = device;
  state_->plugin = &load_plugin(backend);
  const FlagcvBackendV2& plugin = *state_->plugin;
  if (plugin.create_context(device, &state_->context) != 0) {
    state_->plugin = nullptr;  // nothing to release
    throw Error(std::string("flagcv: ") + plugin.last_error());
  }
  char buffer[256];
  if (plugin.device_id(state_->context, buffer, sizeof(buffer)) != 0)
    throw Error(plugin.last_error());
  state_->device_id = buffer;
  if (plugin.target(state_->context, buffer, sizeof(buffer)) != 0)
    throw Error(plugin.last_error());
  state_->target = buffer;
}

Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;

const std::string& Handle::backend() const {
  if (!state_) throw Error("flagcv: handle was moved from");
  return state_->backend;
}

ResolvedKernel resolve_kernel(const std::string& backend,
                              const std::string& kernel) {
  const std::string root = data_dir();
  const struct {
    std::string dir;
    bool platform;
  } tiers[] = {{root + "/backends/" + backend + "/kernels", true},
               {root + "/kernels", false}};

  for (const auto& tier : tiers) {
    const std::string registry_path = tier.dir + "/registry.json";
    std::error_code inspection_error;
    const auto status = std::filesystem::symlink_status(registry_path, inspection_error);
    if (status.type() == std::filesystem::file_type::not_found && tier.platform)
      continue;
    if (inspection_error) {
      throw Error("flagcv: cannot inspect kernel registry " + registry_path +
                  ": " + inspection_error.message());
    }
    json::Value registry = json::Value::parse(read_file(registry_path));
    for (const json::Value& entry : registry.at("kernels").as_array()) {
      if (entry.at("name").as_string() != kernel) continue;
      // Platform registration is exclusive: the first tier that names the
      // kernel wins, and its failures are never retried against the next.
      return ResolvedKernel{tier.dir + "/" + entry.at("source").as_string(),
                            entry.at("function").as_string(), tier.platform};
    }
  }
  throw Error("flagcv: no kernel registered for '" + kernel + "' on backend '" +
              backend + "'");
}

void launch_pointwise(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                      const GpuMat& dst, const GpuMat* mask,
                      PointwiseMode mode, Stream stream, double scale) {
  launch(handle, lower_pointwise(src1, src2, dst, mask, mode, scale), stream);
}

void launch(Handle& handle, LoweredOp op, Stream stream) {
  auto& state = HandleAccess::state(handle);
  const FlagcvBackendV2& plugin = *state.plugin;
  if (state.compiler_identity.empty())
    state.compiler_identity = compiler_identity(state.backend);

  // Per-handle variants share the device, provider and compiler identity.
  // Use scratch storage for the lookup and keep an owned signature on misses.
  std::byte lookup_storage[256];
  std::pmr::monotonic_buffer_resource lookup_memory(lookup_storage, sizeof(lookup_storage));
  std::pmr::vector<std::int64_t> signature(&lookup_memory);
  signature.reserve(op.images.size() + op.params.size());
  const std::string_view kernel_name(op.kernel);
  std::size_t hash = std::hash<std::string_view>{}(kernel_name);
  auto add_signature = [&](std::int64_t value) {
    signature.push_back(value);
    hash ^= std::hash<std::int64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  };
  for (const auto& image : op.images) add_signature(signature_value(image));
  for (const auto& param : op.params) add_signature(signature_value(param));

  auto [it, end] = state.kernels.equal_range(hash);
  for (; it != end; ++it) {
    const auto& candidate = it->second;
    if (candidate.name == kernel_name &&
        std::equal(signature.begin(), signature.end(),
                   candidate.signature.begin(), candidate.signature.end()))
      break;
  }
  if (it == end) {
    const ResolvedKernel resolved = resolve_kernel(state.backend, op.kernel);
    std::ostringstream body;
    body << "\"op\":\"compile\",\"backend\":" << json::quote(state.backend)
         << ",\"target\":" << json::quote(state.target)
         << ",\"kernel\":" << json::quote(op.kernel)
         << ",\"source\":" << json::quote(resolved.source)
         << ",\"function\":" << json::quote(resolved.function)
         << ",\"from_platform_registry\":"
         << (resolved.from_platform_registry ? "true" : "false")
         << ",\"images\":[";
    for (std::size_t i = 0; i < op.images.size(); ++i) {
      const ImageSpec& image = op.images[i];
      body << (i ? "," : "") << "{\"name\":" << json::quote(image.name)
           << ",\"depth\":\"" << depth_name(image.depth)
           << "\",\"channels\":" << image.channels
           << ",\"aligned16\":" << (aligned16(image.address) ? "true" : "false")
           << "}";
    }
    body << "],\"params\":[";
    for (std::size_t i = 0; i < op.params.size(); ++i) {
      const auto& param = op.params[i];
      body << (i ? "," : "") << "{\"name\":" << json::quote(param.name);
      if (param.compile_time)
        body << ",\"type\":\"constexpr\",\"value\":" << param.value << "}";
      else
        body << ",\"type\":\"" << runtime_type(param.value)
             << "\",\"div16\":" << (divisible16(param) ? "true" : "false")
             << "}";
    }
    body << "]";

    json::Value reply = CompilerClient::instance().request(body.str());
    const std::string binary = read_file(reply.at("artifact").as_string());

    LoadedKernel loaded;
    loaded.name = kernel_name;
    loaded.signature.assign(signature.begin(), signature.end());
    const std::int64_t warp_size =
        std::stoll(state.target.substr(state.target.rfind(':') + 1));
    loaded.block_threads =
        static_cast<unsigned int>(reply.at("num_warps").as_int() * warp_size);
    loaded.shared_bytes =
        static_cast<unsigned int>(reply.at("shared").as_int());
    const auto& grid = reply.at("grid").as_array();
    if (grid.size() != 3)
      throw Error("flagcv: compiler returned a grid without three dimensions");
    for (int d = 0; d < 3; ++d) {
      if (grid[d].is_number()) {
        loaded.grid[d].fixed = grid[d].as_int();
        continue;
      }
      for (const auto& item : grid[d].at("product").as_array()) {
        const auto name = item.as_string();
        auto param = std::find_if(op.params.begin(), op.params.end(),
                                  [&](const auto& p) { return p.name == name; });
        if (param == op.params.end())
          throw Error("flagcv: grid refers to unknown param '" + name + "'");
        loaded.grid[d].product.push_back(std::distance(op.params.begin(), param));
      }
      loaded.grid[d].block = grid[d].at("block").as_int();
      if (grid[d].has("limit")) loaded.grid[d].limit = grid[d].at("limit").as_int();
    }
    const unsigned int flags =
        reply.has("load_flags")
            ? static_cast<unsigned int>(reply.at("load_flags").as_int())
            : 0u;
    const auto function = reply.at("function").as_string();
    it = state.kernels.emplace(hash, std::move(loaded));
    if (plugin.load(state.context, binary.data(), binary.size(), function.c_str(),
                    it->second.shared_bytes, flags, &it->second.kernel) != 0) {
      state.kernels.erase(it);
      throw Error(plugin.last_error());
    }
  }
  const LoadedKernel& kernel = it->second;

  unsigned int grid[3];
  for (int d = 0; d < 3; ++d) {
    std::int64_t extent = kernel.grid[d].fixed;
    if (!kernel.grid[d].product.empty()) {
      std::int64_t total = 1;
      for (const auto index : kernel.grid[d].product) {
        const auto value = op.params[index].value;
        if (value <= 0 || total > std::numeric_limits<std::int64_t>::max() / value)
          throw Error("flagcv: invalid launch grid parameter");
        total *= value;
      }
      const auto block = kernel.grid[d].block;
      if (block <= 0) throw Error("flagcv: invalid launch grid block size");
      extent = std::min(total / block + (total % block != 0),
                        kernel.grid[d].limit);
    }
    if (extent <= 0 || extent > std::numeric_limits<unsigned int>::max()) {
      throw Error("flagcv: launch grid dimension out of range");
    }
    grid[d] = static_cast<unsigned int>(extent);
  }

  // Native launch convention: one pointer per kernel argument, to its value.
  // `kinds` records each argument's width in the same order: 0 for an 8-byte
  // value (device pointer or i64), 1 for a 4-byte one (i32). Only Ascend's
  // packed launch needs it; the others launch from the pointer array alone.
  std::byte argument_storage[640];
  std::pmr::monotonic_buffer_resource argument_memory(argument_storage, sizeof(argument_storage));
  std::pmr::vector<std::int64_t> wide(&argument_memory);
  std::pmr::vector<std::int32_t> narrow(&argument_memory);
  std::pmr::vector<void*> args(&argument_memory);
  std::pmr::vector<unsigned int> kinds(&argument_memory);
  wide.reserve(op.params.size());
  narrow.reserve(op.params.size());
  args.reserve(op.images.size() + op.params.size() + 2);
  kinds.reserve(args.capacity());
  for (auto& image : op.images) {
    args.push_back(&image.address);
    kinds.push_back(0);
  }
  for (const auto& param : op.params) {
    if (param.compile_time) continue;  // baked into the binary, not an argument
    if (fits_i32(param.value)) {
      narrow.push_back(static_cast<std::int32_t>(param.value));
      args.push_back(&narrow.back());
      kinds.push_back(1);
    } else {
      wide.push_back(param.value);
      args.push_back(&wide.back());
      kinds.push_back(0);
    }
  }
  std::uint64_t global_scratch = 0, profile_scratch = 0;
  if (plugin.appends_scratch_args) {
    // Triton's ABI ends with two scratch pointers. The compiler rejects any
    // kernel that needs real scratch memory, so null is correct here.
    args.push_back(&global_scratch);
    args.push_back(&profile_scratch);
    kinds.push_back(0);
    kinds.push_back(0);
  }

  if (plugin.launch(state.context, kernel.kernel, grid, kernel.block_threads,
                    kernel.shared_bytes, stream, args.data(), kinds.data(),
                    static_cast<unsigned int>(args.size())) != 0) {
    throw Error(plugin.last_error());
  }
}

}  // namespace flagcv
