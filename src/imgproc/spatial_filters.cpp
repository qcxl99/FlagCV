/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "src/imgproc/gaussian_exp.hpp"
#include "src/imgproc/spatial_filters.hpp"
#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

std::uint64_t coefficient_magnitude_bits(double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559);
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits & 0x7fffffffffffffffULL;
}

// The table is uploaded as raw double. Triton narrows its values with RNE,
// independently of the host's rounding and DAZ/FTZ modes. This certificate
// conservatively excludes any nonzero value below the normal F32 range.
bool normal_f32_significand(std::uint64_t bits, std::uint32_t& significand, int& exponent) {
  constexpr auto fraction_mask = (std::uint64_t{1} << 52) - 1;
  const int binade = static_cast<int>(bits >> 52) - 1023;
  if (binade < -126 || binade > 127) return false;
  constexpr int dropped = 29;
  constexpr auto halfway = std::uint64_t{1} << (dropped - 1);
  const auto mantissa = (bits & fraction_mask) | (std::uint64_t{1} << 52);
  const auto remainder = mantissa & ((std::uint64_t{1} << dropped) - 1);
  significand = static_cast<std::uint32_t>(mantissa >> dropped);
  significand += remainder > halfway || (remainder == halfway && (significand & 1));
  exponent = binade - 23;
  // A rounded significand can have the carry bit set. The bound below
  // canonicalizes it along with other trailing zero bits.
  return binade != 127 || significand < (1u << 24);
}

// Use the original coefficient for the exact-integer shortcut. In particular,
// a raw tiny nonzero double must never be admitted as zero under host DAZ.
int small_integer_magnitude(double value) {
  const auto bits = coefficient_magnitude_bits(value);
  if (!bits) return 0;
  const int exponent = static_cast<int>(bits >> 52) - 1023;
  if (exponent < 0 || exponent > 8) return -1;
  const auto mantissa = (bits & ((std::uint64_t{1} << 52) - 1)) | (std::uint64_t{1} << 52);
  const int shift = 52 - exponent;
  if (mantissa & ((std::uint64_t{1} << shift) - 1)) return -1;
  return static_cast<int>(mantissa >> shift);
}

// Bound the numerator of every partial sum in a common binary unit.
class U16CorrelationBound {
 public:
  void add(double value, std::uint32_t multiplier) {
    const auto bits = coefficient_magnitude_bits(value);
    if (!normal_ || bits == 0) return;
    std::uint32_t significand{};
    int exponent{};
    if (!normal_f32_significand(bits, significand, exponent)) {
      exact_ = normal_ = false;
      return;
    }
    while ((significand & 1) == 0) {
      significand >>= 1;
      ++exponent;
    }
    auto term = std::uint64_t{significand} * multiplier;
    min_exponent_ = std::min(min_exponent_, exponent);
    max_bit_ = std::max(max_bit_, exponent + std::ilogb(static_cast<double>(term)));
    ++terms_;
    if (!exact_) return;
    if (units_ == 0) unit_exponent_ = exponent;
    const auto unit = std::min(exponent, unit_exponent_);
    constexpr std::uint64_t limit = (std::uint64_t{1} << 53) - 1;
    const int old_shift = unit_exponent_ - unit;
    const int new_shift = exponent - unit;
    if (old_shift >= 53 || new_shift >= 53 || units_ > (limit >> old_shift) ||
        term > (limit >> new_shift)) {
      exact_ = false;
      return;
    }
    units_ <<= old_shift;
    term <<= new_shift;
    if (term > limit - units_) {
      exact_ = false;
      return;
    }
    units_ += term;
    unit_exponent_ = unit;
  }

  bool exact() const { return exact_; }
  bool pair_exact() const {
    if (!normal_) return false;
    if (terms_ <= 1) return true;
    const int log_terms = 1 + std::ilogb(static_cast<double>(terms_ - 1));
    // Bound the absolute sum by 2^(max_bit - min_exponent + 1 + log_terms).
    // The low accumulator stores at most terms_ errors, each at most
    // 2^(absolute_sum_bits - 53) in units of the smallest binary exponent.
    return max_bit_ - min_exponent_ + 1 + 2 * log_terms <= 106;
  }
  int integer_unit() const {
    constexpr int unavailable = std::numeric_limits<int>::max();
    if (!normal_ || terms_ == 0) return unavailable;
    const int log_terms = terms_ == 1 ? 0 : 1 + std::ilogb(static_cast<double>(terms_ - 1));
    // The signed three-word accumulator must represent every prefix, not only
    // the final sum. Each term contributes strictly less than 2^(max_bit+1).
    int bits = max_bit_ - min_exponent_ + 1 + log_terms;
    return bits <= 191 ? min_exponent_ : unavailable;
  }

 private:
  std::uint64_t units_ = 0;
  int unit_exponent_ = 0;
  bool exact_ = true;
  bool normal_ = true;
  std::uint32_t terms_ = 0;
  int min_exponent_ = std::numeric_limits<int>::max();
  int max_bit_ = std::numeric_limits<int>::min();
};

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
  bool all_normal_f32_taps = true;
  const bool has_nonzero_delta = coefficient_magnitude_bits(options.delta) != 0;
  bool fp32_integer_exact = source == Depth::k16U && destination == Depth::k16U && !has_nonzero_delta;
  // A U16 product and each partial sum are exact integers in F32 when
  // the absolute tap sum multiplied by 65535 fits in the F32 significand.
  constexpr auto max_exact_u16_tap_sum = (std::uint32_t{1} << std::numeric_limits<float>::digits) /
                                          std::numeric_limits<std::uint16_t>::max();
  std::uint32_t absolute_tap_sum = 0;
  const bool bound_u16 = source == Depth::k16U && !work_double && count >= 50;
  U16CorrelationBound bound;
  if (bound_u16 && std::abs(options.delta) >= std::numeric_limits<double>::epsilon())
    bound.add(options.delta, 1);
  for (int i = 0; i < count; ++i) {
    check_coefficient(coefficients[i], work_double);
    table.push_back(coefficients[i]);
    const double effective = work_double ? coefficients[i] : static_cast<float>(coefficients[i]);
    zero_filter &= effective == 0.0;
    // The row-pair kernel omits the device zero test.
    all_normal_f32_taps &= std::abs(coefficients[i]) >= std::numeric_limits<float>::min();
    if (bound_u16) bound.add(coefficients[i], std::numeric_limits<std::uint16_t>::max());
    if (fp32_integer_exact) {
      const auto magnitude = small_integer_magnitude(coefficients[i]);
      fp32_integer_exact = magnitude >= 0 &&
                           static_cast<std::uint32_t>(magnitude) <= max_exact_u16_tap_sum - absolute_tap_sum;
      if (fp32_integer_exact) absolute_tap_sum += static_cast<std::uint32_t>(magnitude);
    }
  }
  data_ = std::make_shared<Data>(Data{kernel, source, destination, options.border, anchor,
                                     zero_filter, all_normal_f32_taps, fp32_integer_exact, bound_u16 && bound.exact(),
                                     bound_u16 && bound.pair_exact(), has_nonzero_delta,
                                     bound_u16 ? bound.integer_unit() : std::numeric_limits<int>::max(),
                                     std::move(table)});
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
