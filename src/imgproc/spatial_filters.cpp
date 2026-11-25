/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "src/imgproc/gaussian_exp.hpp"
#include "src/imgproc/spatial_filters.hpp"
#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace flagcv {
namespace {

constexpr int kMaxGaussianAxis = 65535;

void check_border(Border border) {
  switch (border) {
    case Border::kConstant:
    case Border::kReplicate:
    case Border::kReflect:
    case Border::kReflect101:
      return;
    default:
      throw std::invalid_argument("spatial filter border must be CONSTANT, REPLICATE, REFLECT or REFLECT_101");
  }
}

int gaussian_axis(int size, double sigma, Depth depth, int source_extent, Border border) {
  if (source_extent == 1 && border != Border::kConstant)
    return 1;
  if (size <= 0 && sigma > 0) {
    const double diameter = sigma * (depth == Depth::k8U ? 6.0 : 8.0) + 1.0;
    if (diameter > kMaxGaussianAxis + 0.5)
      throw std::invalid_argument("GaussianBlur automatic kernel exceeds the supported size");
    size = spatial::nearest_even(diameter) | 1;
  }
  if (size < 1 || size > kMaxGaussianAxis || size % 2 == 0)
    throw std::invalid_argument("GaussianBlur kernel axes must be positive, odd and at most 65535");
  return size;
}

std::vector<double> gaussian_weights(int n, double sigma) {
  if (n == 1)
    return {1.0};
  if (sigma <= 0) {
    switch (n) {
      case 3: return {0.25, 0.5, 0.25};
      case 5: return {1./16, 4./16, 6./16, 4./16, 1./16};
      case 7: return {2./64, 7./64, 14./64, 18./64, 14./64, 7./64, 2./64};
      case 9: return {4./256, 13./256, 30./256, 51./256, 60./256,
                      51./256, 30./256, 13./256, 4./256};
      default: sigma = std::fma(0.15, n, 0.35); break;
    }
  }

  std::vector<double> coefficients(n);
  const int center = n / 2;
  const double scale = -0.125 / (sigma * sigma);
  double side_sum = 0;
  for (int i = 0; i < center; ++i) {
    const double distance = 2. * (center - i);
    const double weight = spatial::gaussian_exp(distance * distance * scale);
    coefficients[i] = weight;
    side_sum += weight;
  }
  const double inverse = 1.0 / (2.0 * side_sum + 1.0);
  for (int i = 0; i < center; ++i) {
    coefficients[i] *= inverse;
    coefficients[n - i - 1] = coefficients[i];
  }
  coefficients[center] = inverse;
  return coefficients;
}

void quantize(std::vector<double>& coefficients, int bits) {
  const int scale = 1 << bits;
  const auto center = coefficients.size() / 2;
  double residue = 0;
  int side_sum = 0;
  for (std::size_t i = 0; i < center; ++i) {
    const double adjusted = coefficients[i] * scale + residue;
    const int weight = spatial::nearest_even(adjusted);
    residue = adjusted - weight;
    coefficients[i] = coefficients[coefficients.size() - 1 - i] = weight;
    side_sum += weight;
  }
  coefficients[center] = scale - 2 * side_sum;
}

bool correlation_depths(Depth source, Depth destination) {
  switch (source) {
    case Depth::k8U:
      return destination == Depth::k8U || destination == Depth::k16U ||
          destination == Depth::k16S || destination == Depth::k32F || destination == Depth::k64F;
    case Depth::k16U:
    case Depth::k16S:
      return destination == source || destination == Depth::k32F || destination == Depth::k64F;
    case Depth::k32F:
    case Depth::k64F:
      return destination == source;
    default:
      return false;
  }
}

void check_coefficient(double value, bool work_double) {
  if (!std::isfinite(value) ||
      (!work_double && std::abs(value) > std::numeric_limits<float>::max()))
    throw std::invalid_argument("filter2D coefficient or delta is outside working precision");
}

}  // namespace

GaussianPlan::GaussianPlan(Size source, Depth depth, GaussianOptions options) {
  if (source.width <= 0 || source.height <= 0 ||
      (depth != Depth::k8U && depth != Depth::k16U && depth != Depth::k32F && depth != Depth::k64F))
    throw std::invalid_argument("GaussianBlur needs positive source extents and U8/U16/F32/F64");
  if (!std::isfinite(options.sigma_x) || !std::isfinite(options.sigma_y))
    throw std::invalid_argument("GaussianBlur sigma must be finite");
  check_border(options.border);
  const auto sx = options.sigma_x;
  const auto sy = options.sigma_y > 0 ? options.sigma_y : sx;
  const Size kernel{gaussian_axis(options.kernel.width, sx, depth, source.width, options.border),
                    gaussian_axis(options.kernel.height, sy, depth, source.height, options.border)};
  // OpenCV resolves both axes, clamps their sigmas, then shares the
  // horizontal coefficients for equal axes and nearly identical sigmas.
  const bool share_axis = kernel.width == kernel.height &&
      std::abs(std::max(sx, 0.) - std::max(sy, 0.)) < std::numeric_limits<double>::epsilon();
  auto x = gaussian_weights(kernel.width, sx);
  auto y = share_axis ? x : gaussian_weights(kernel.height, sy);
  if (depth == Depth::k8U || depth == Depth::k16U) {
    const int bits = depth == Depth::k8U ? 8 : 16;
    quantize(x, bits);
    quantize(y, bits);
  }
  x.insert(x.end(), y.begin(), y.end());
  data_ = std::make_shared<Data>(Data{source, kernel, depth, options.border, std::move(x)});
}

Size GaussianPlan::kernel_size() const { return data_->kernel; }

Depth GaussianPlan::scratch_depth() const {
  return is_integer(data_->depth) ? Depth::k32S : data_->depth;
}

HostTableData GaussianPlan::table() const {
  return {data_->coefficients.data(), data_->coefficients.size() * sizeof(double)};
}

Filter2DPlan::Filter2DPlan(Size kernel, const double* coefficients, Depth source, Filter2DOptions options) {
  if (!coefficients || kernel.width <= 0 || kernel.height <= 0 ||
      std::int64_t{kernel.width} * kernel.height >= std::numeric_limits<int>::max())
    throw std::invalid_argument("filter2D needs a positive coefficient matrix with an int32 addressable tap count");
  const Depth destination = options.ddepth.value_or(source);
  if (!correlation_depths(source, destination))
    throw std::invalid_argument("filter2D source and destination depth combination is unsupported");
  check_border(options.border);
  auto anchor = options.anchor;
  if (anchor.x == -1) anchor.x = kernel.width / 2;
  if (anchor.y == -1) anchor.y = kernel.height / 2;
  if (anchor.x < 0 || anchor.x >= kernel.width || anchor.y < 0 || anchor.y >= kernel.height)
    throw std::invalid_argument("filter2D anchor is outside the coefficient matrix");
  const bool work_double = destination == Depth::k64F;
  check_coefficient(options.delta, work_double);
  const auto count = kernel.width * kernel.height;
  std::vector<double> table;
  table.reserve(count + 1);
  table.push_back(options.delta);
  bool zero_filter = true;
  bool dense_filter = true;
  for (int i = 0; i < count; ++i) {
    check_coefficient(coefficients[i], work_double);
    table.push_back(coefficients[i]);
    const bool nonzero = work_double ? coefficients[i] != 0.0 :
                                      static_cast<float>(coefficients[i]) != 0.0f;
    zero_filter &= !nonzero;
    dense_filter &= nonzero;
  }
  data_ = std::make_shared<Data>(Data{kernel, source, destination, options.border, anchor,
                                    zero_filter, dense_filter, std::move(table)});
}

HostTableData Filter2DPlan::table() const {
  return {data_->coefficients.data(), data_->coefficients.size() * sizeof(double)};
}

void GaussianBlur(Handle& handle, const GpuMat& src, const GpuMat& dst,
                  const GaussianPlan& plan, DeviceWorkspace workspace, Stream stream) {
  launch(handle, lower_gaussian(src, dst, plan, workspace), stream);
}

void GaussianBlur(Handle& handle, const GpuMat& src, const GpuMat& dst,
                  const GaussianPlan& plan, DeviceWorkspace workspace,
                  const GpuMat& scratch, Stream stream) {
  const auto size = plan.kernel_size();
  if (size.width == 1 && size.height == 1) {
    GaussianBlur(handle, src, dst, plan, workspace, stream);
    return;
  }
  auto passes = lower_gaussian_separable(src, dst, plan, workspace, scratch);
  launch(handle, passes[0], stream);
  launch(handle, passes[1], stream);
}

void filter2D(Handle& handle, const GpuMat& src, const GpuMat& dst,
              const Filter2DPlan& plan, DeviceWorkspace workspace, Stream stream) {
  launch(handle, lower_filter2d(src, dst, plan, workspace), stream);
}

}  // namespace flagcv
