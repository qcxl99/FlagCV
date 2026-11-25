// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "src/runtime/runtime.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    auto pattern = (fs::temp_directory_path() / "flagcv-registry-XXXXXX").string();
    const char* root = mkdtemp(pattern.data());
    if (!root) throw std::runtime_error("mkdtemp failed");
    path = root;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path, error);
  }
  fs::path path;
};

void write(const fs::path& path, const char* document) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path);
  stream << document;
  require(static_cast<bool>(stream), "test registry write failed");
}

void exercise_registries() {
  TemporaryDirectory root;
  const auto common = root.path / "kernels/registry.json";
  const auto platform = root.path / "backends/nvidia/kernels/registry.json";
  require(setenv("FLAGCV_DATA_DIR", root.path.c_str(), 1) == 0, "setenv failed");
  write(common, R"({"kernels":[{"name":"add","source":"add.py","function":"base"}]})");
  auto common_kernel = flagcv::resolve_kernel("nvidia", "add");
  require(!common_kernel.from_platform_registry && common_kernel.function == "base",
          "missing platform registry must select common implementation");
  write(platform, R"({"kernels":[{"name":"flip","source":"flip.py","function":"flip"}]})");
  require(!flagcv::resolve_kernel("nvidia", "add").from_platform_registry,
          "unregistered operation must select common implementation");
  write(platform, R"({"kernels":[{"name":"add","source":"missing.py","function":"override"}]})");
  const auto selected = flagcv::resolve_kernel("nvidia", "add");
  require(selected.from_platform_registry && selected.function == "override",
          "named platform entry must take precedence even if its source is missing");
  write(platform, "broken json");
  bool failed = false;
  try {
    flagcv::resolve_kernel("nvidia", "add");
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, "invalid platform registry must fail without falling back");
}
}  // namespace

int main() {
  try {
    exercise_registries();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
