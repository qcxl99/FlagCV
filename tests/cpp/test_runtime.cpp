// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
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

void convenience(const std::filesystem::path& directory) {
  const auto backend = directory / "backend.trace";
  const auto compiler = directory / "compiler.trace";
  std::ofstream{backend};
  std::ofstream{compiler};
  alignas(16) std::array<unsigned char, 128> src{}, dst{};
  const flagcv::GpuMat a(src.data(), 16, 2, flagcv::Depth::k8U, 1, 64);
  const flagcv::GpuMat out(dst.data(), 16, 2, flagcv::Depth::k8U, 1, 64);
  auto add = [&] { flagcv::add(a, a, out); };
  auto require_unselected = [&] {
    try {
      add();
    } catch (const flagcv::Error& error) {
      require(std::string(error.what()).find("setDevice()") != std::string::npos,
              "missing setDevice diagnostic");
      return;
    }
    throw std::runtime_error("used a default handle without selecting a device");
  };
  require_unselected();
  std::async(std::launch::async, [&] {
    flagcv::setDevice("testfixture");
    add();
    flagcv::setDevice("testfixture", 0);
    add();
    flagcv::setDevice("testfixture", 1);
    add();
    try {
      flagcv::setDevice("testfixture", -1);
      throw std::runtime_error("invalid device accepted");
    } catch (const flagcv::Error&) {
      add();
    }
    flagcv::setDevice("testfixture");
    std::async(std::launch::async, [&] {
      require_unselected();
      flagcv::Handle explicit_handle("testfixture", 1);
      flagcv::add(explicit_handle, a, a, out);
      require_unselected();
      flagcv::setDevice("testfixture");
      add();
      flagcv::add(explicit_handle, a, a, out);
      add();
    }).get();
    add();
  }).get();
  require_unselected();
  check_trace(lines(compiler), std::vector<std::string>(4, "add platform platform_add"),
              "selected devices did not reuse their compiler cache");
  const auto events = lines(backend);
  std::vector<std::string> launches;
  std::copy_if(events.begin(), events.end(), std::back_inserter(launches),
               [](const std::string& line) { return line.rfind("launch ", 0) == 0; });
  const std::string zero = "launch 0 platform_add 1 default";
  const std::string one = "launch 1 platform_add 1 default";
  check_trace(launches, {zero, zero, one, one, one, zero, one, zero, zero},
              "selected devices, threads or explicit handles interfered");
  require(std::count(events.begin(), events.end(), "create 0") == 2 &&
          std::count(events.begin(), events.end(), "create 1") == 2 &&
          std::count(events.begin(), events.end(), "destroy 0") == 2 &&
          std::count(events.begin(), events.end(), "destroy 1") == 2,
          "device selection did not create and release exactly one handle per thread and key");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    try {
      flagcv::Handle legacy("hip");
      throw std::logic_error("the removed HIP backend was accepted");
    } catch (const flagcv::Error& error) {
      if (std::string(error.what()).find("select amd (ROCm) or hygon (DTK)") == std::string::npos)
        throw;
    }

    require(argc == 2, "expected fixture directory");
    exercise(argv[1]);
    convenience(argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
