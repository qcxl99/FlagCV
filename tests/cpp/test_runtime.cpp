// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* error) {
  if (!condition) throw std::runtime_error(error);
}

std::vector<std::string> lines(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::vector<std::string> result;
  for (std::string line; std::getline(stream, line);) result.push_back(line);
  return result;
}

void check_trace(const std::vector<std::string>& actual,
                 const std::vector<std::string>& expected, const char* label) {
  if (actual == expected) return;
  for (const auto& line : actual) std::cerr << line << '\n';
  throw std::runtime_error(label);
}

void exercise(const std::filesystem::path& directory) {
  const auto backend = directory / "backend.trace";
  const auto compiler = directory / "compiler.trace";
  std::ofstream{backend};
  std::ofstream{compiler};
  require(setenv("FLAGCV_TEST_BACKEND_TRACE", backend.c_str(), 1) == 0, "backend trace setup");
  require(setenv("FLAGCV_TEST_COMPILER_TRACE", compiler.c_str(), 1) == 0, "compiler trace setup");

  alignas(16) std::array<unsigned char, 128> a{}, b{}, out{};
  int fake_stream = 0;
  auto view = [](void* ptr, int width) {
    return flagcv::GpuMat(ptr, width, 2, flagcv::Depth::k8U, 1, 64);
  };
  auto add = [&](flagcv::Handle& handle, int width) {
    flagcv::add(handle, view(a.data(), width), view(b.data(), width),
                view(out.data(), width), &fake_stream);
  };
  {
    flagcv::Handle survivor("testfixture", 1);
    {
      flagcv::Handle first("testfixture", 0);
      add(first, 16);  // cold path
      add(first, 32);  // same specialization, distinct grid
      add(first, 17);  // divisibility changes specialization
      add(survivor, 16);
    }
    add(survivor, 32);
    flagcv::bitwise_not(survivor, view(a.data(), 16), view(out.data(), 16));
  }
  const auto compiles = lines(compiler);
  require(compiles.size() == 4, "unexpected number of compiler requests");
  require(std::count(compiles.begin(), compiles.end(), "add platform platform_add") == 3,
          "platform registration did not take priority or compiled the wrong number of times");
  require(compiles.back().rfind("bitwise_not common ", 0) == 0,
          "missing platform registration did not select the common kernel");
  const auto not_symbol = compiles.back().substr(std::string("bitwise_not common ").size());
  auto events = lines(backend);
  // At destruction the cache does not specify the order between modules.
  const auto unload = std::find(events.begin(), events.end(), "unload 1 platform_add");
  if (unload != events.end() && unload != events.begin() && *(unload - 1) == "unload 1 " + not_symbol)
    std::iter_swap(unload, unload - 1);
  check_trace(events, {"create 1", "create 0", "load 0 platform_add",
                       "launch 0 platform_add 1 custom", "launch 0 platform_add 2 custom",
                       "load 0 platform_add", "launch 0 platform_add 2 custom",
                       "load 1 platform_add", "launch 1 platform_add 1 custom",
                       "unload 0 platform_add", "unload 0 platform_add", "destroy 0",
                       "launch 1 platform_add 2 custom", "load 1 " + not_symbol,
                       "launch 1 " + not_symbol + " 1 default",
                       "unload 1 platform_add", "unload 1 " + not_symbol, "destroy 1"},
              "unexpected load/launch/unload/context sequence");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    exercise(argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
