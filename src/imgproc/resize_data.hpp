/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <flagcv/imgproc.hpp>

#include "resize_tables.hpp"

#include <cstring>
#include <vector>

namespace flagcv {

struct ResizePlan::Data {
  struct Table {
    const char* name;
    Depth depth;
    std::size_t offset;
  };

  Size source;
  Size destination;
  Size requested_size;
  Depth depth;
  int channels;
  resize_tables::RouteInfo route;
  int xmax = 0;
  int kx = 0;
  int ky = 0;
  std::vector<std::uint32_t> words;
  std::vector<Table> tables;

  template<class Value>
  void append(const char* name, Depth type, const std::vector<Value>& values) {
    static_assert(sizeof(Value) == sizeof(std::uint32_t));
    tables.push_back({name, type, words.size() * sizeof(std::uint32_t)});
    const auto first = words.size();
    words.insert(words.end(), values.size(), 0);
    if (!values.empty())
      std::memcpy(words.data() + first, values.data(), values.size() * sizeof(Value));
  }
};

struct ResizeAccess {
  static const ResizePlan::Data& data(const ResizePlan& plan) { return *plan.data_; }
};

}  // namespace flagcv
