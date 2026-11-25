// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  flagcv::Handle handle("testfixture", 0);
  alignas(16) std::array<unsigned char, 16> a{}, b{};
  const flagcv::GpuMat src(a.data(), 8, 2, flagcv::Depth::k8U, 1);
  const flagcv::GpuMat dst(b.data(), 8, 2, flagcv::Depth::k8U, 1);
  for (std::string_view mode : {"missing", "wrong", "valid"}) {
    std::ofstream(std::string(argv[1]) + "/scratch-mode") << mode;
    try {
      flagcv::flip(handle, src, dst, 1);
    } catch (const flagcv::Error& error) {
      const std::string_view expected = mode == "valid"
          ? "reached module load" : "compiler scratch ABI does not match";
      if (std::string_view(error.what()).find(expected) != std::string_view::npos) continue;
      std::cerr << mode << ": " << error.what() << '\n';
      return 1;
    }
    std::cerr << mode << ": expected a controlled rejection\n";
    return 1;
  }
  return 0;
}
