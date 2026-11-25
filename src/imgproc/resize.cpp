/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "resize_data.hpp"
#include "src/runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flagcv {
namespace {

int mode(Interpolation interpolation) {
  switch (interpolation) {
    case Interpolation::kNearest: return resize_tables::INTER_NEAREST;
    case Interpolation::kLinear: return resize_tables::INTER_LINEAR;
    case Interpolation::kCubic: return resize_tables::INTER_CUBIC;
    case Interpolation::kArea: return resize_tables::INTER_AREA;
  }
  throw std::invalid_argument("resize unsupported interpolation");
}

void validate_size(Size size, int channels) {
  constexpr int limit = std::numeric_limits<std::int32_t>::max();
  if (size.width <= 0 || size.height <= 0)
    throw std::invalid_argument("resize dimensions must be positive");
  const auto row = std::int64_t{size.width} * channels;
  if (row > limit || size.height > limit / row)
    throw std::invalid_argument("resize image exceeds int32 element indexing");
}

std::vector<int> nearest_offsets(int dst, int src, double inverse) {
  const double scale = 1.0 / inverse;
  std::vector<int> offsets(dst);
  for (int i = 0; i < dst; ++i)
    offsets[i] = static_cast<int>(std::min(std::floor(i * scale), double(src - 1)));
  return offsets;
}

bool matches_area_polyphase(const resize_tables::AreaTab& table, int source,
                            int destination, double scale) {
  if (scale != 1.5 || destination % 2 != 0 || source != 3 * (destination / 2) ||
      table.taps != 2)
    return false;
  constexpr float two_thirds = 2.0f / 3.0f;
  constexpr float one_third = 1.0f / 3.0f;
  for (int output = 0; output < destination; ++output) {
    const int first = 3 * (output / 2) + output % 2;
    const int tap = output * 2;
    const bool odd = output % 2 != 0;
    if (table.count[output] != 2 || table.src_index[tap] != first ||
        table.src_index[tap + 1] != first + 1 ||
        table.alpha[tap] != (odd ? one_third : two_thirds) ||
        table.alpha[tap + 1] != (odd ? two_thirds : one_third))
      return false;
  }
  return true;
}

}  // namespace

ResizePlan::ResizePlan(Size source, Depth depth, int channels, ResizeOptions options) {
  using namespace resize_tables;
  if ((depth != Depth::k8U && depth != Depth::k32F) ||
      (channels != 1 && channels != 3 && channels != 4))
    throw std::invalid_argument("resize supports 8U/32F and 1, 3 or 4 channels");
  const auto interpolation = mode(options.interpolation);
  validate_size(source, channels);
  if (options.size.width <= 0 || options.size.height <= 0) {
    constexpr double limit = std::numeric_limits<std::int32_t>::max();
    if (!std::isfinite(options.fx) || !std::isfinite(options.fy) ||
        options.fx <= 0 || options.fy <= 0 ||
        source.width * options.fx > limit || source.height * options.fy > limit)
      throw std::invalid_argument("resize scales must be finite, positive and within range");
  }
  const auto size = resolve_size(source.height, source.width, options.size.width,
                                 options.size.height, options.fx, options.fy);
  const Size destination{size.dst_w, size.dst_h};
  validate_size(destination, channels);
  auto data = std::make_shared<Data>();
  data->source = source;
  data->destination = destination;
  data->requested_size = options.size;
  data->depth = depth;
  data->channels = channels;
  data->route = route(source.height, source.width, destination.height, destination.width,
                      size.inv_y, size.inv_x, interpolation);
  switch (data->route.kind) {
    case ROUTE_COPY:
    case ROUTE_AREA_FAST:
      break;
    case ROUTE_NEAREST:
      data->append("xofs", Depth::k32S, nearest_offsets(destination.width, source.width, size.inv_x));
      data->append("yofs", Depth::k32S, nearest_offsets(destination.height, source.height, size.inv_y));
      break;
    case ROUTE_LINEAR:
    case ROUTE_CUBIC: {
      const int taps = data->route.kind == ROUTE_CUBIC ? 4 : 2;
      const auto x = generic_axis(destination.width, source.width, size.inv_x, taps,
                                  data->route.area_mode, true);
      const auto y = generic_axis(destination.height, source.height, size.inv_y, taps,
                                  data->route.area_mode, false);
      data->xmax = x.xmax;
      data->append("xofs", Depth::k32S, x.ofs);
      if (depth == Depth::k8U) {
        const auto by = fixed_coeffs(y.coeffs);
        std::vector<float> byf;
        byf.reserve(by.size());
        for (int value : by) byf.push_back(static_cast<float>(value) / (2048 * 2048));
        data->append("ax", Depth::k32S, fixed_coeffs(x.coeffs));
        data->append("yofs", Depth::k32S, y.ofs);
        data->append("by", Depth::k32S, by);
        data->append("byf", Depth::k32F, byf);
      } else {
        data->append("ax", Depth::k32F, x.coeffs);
        data->append("yofs", Depth::k32S, y.ofs);
        data->append("by", Depth::k32F, y.coeffs);
        auto alias = data->tables.back();
        alias.name = "byf";
        data->tables.push_back(alias);
      }
      break;
    }
    case ROUTE_AREA: {
      const auto x = area_tab(source.width, destination.width, 1.0 / size.inv_x);
      const auto y = area_tab(source.height, destination.height, 1.0 / size.inv_y);
      data->kx = x.taps;
      data->ky = y.taps;
      data->area_polyphase = channels == 4 &&
          matches_area_polyphase(x, source.width, destination.width, 1.0 / size.inv_x) &&
          matches_area_polyphase(y, source.height, destination.height, 1.0 / size.inv_y);
      data->append("xs", Depth::k32S, x.src_index);
      data->append("xa", Depth::k32F, x.alpha);
      data->append("xc", Depth::k32S, x.count);
      data->append("ys", Depth::k32S, y.src_index);
      data->append("ya", Depth::k32F, y.alpha);
      data->append("yc", Depth::k32S, y.count);
      break;
    }
  }
  data_ = std::move(data);
}

Size ResizePlan::destination_size() const { return data_->destination; }
const void* ResizePlan::table_data() const { return data_->words.data(); }
std::size_t ResizePlan::table_bytes() const { return data_->words.size() * sizeof(std::uint32_t); }

void resize(Handle& handle, const GpuMat& src, const GpuMat& dst,
            const ResizePlan& plan, DeviceWorkspace workspace, Stream stream) {
  launch(handle, lower_resize(src, dst, plan, workspace, 1, 0, 0, false), stream);
}

void resizeBatch(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 int count, std::int64_t src_batch_pitch, std::int64_t dst_batch_pitch,
                 const ResizePlan& plan, DeviceWorkspace workspace, Stream stream) {
  launch(handle, lower_resize(src, dst, plan, workspace, count,
                              src_batch_pitch, dst_batch_pitch, true), stream);
}

}  // namespace flagcv
