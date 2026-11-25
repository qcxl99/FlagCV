// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

#include "cuda_image.hpp"

namespace {
using flagcv::AffineMatrix;
using flagcv::Border;
using flagcv::BorderValue;
using flagcv::Interpolation;
using flagcv_test::bytes_equal;
using flagcv_test::cuda_check;
using flagcv_test::Image;
using flagcv_test::require;
using flagcv_test::Stream;

struct Mode {
  Interpolation flagcv;
  int opencv;
};

struct BorderMode {
  Border flagcv;
  int opencv;
};

constexpr Mode modes[] = {{Interpolation::kNearest, cv::INTER_NEAREST},
                          {Interpolation::kLinear, cv::INTER_LINEAR},
                          {Interpolation::kCubic, cv::INTER_CUBIC},
                          {Interpolation::kArea, cv::INTER_AREA}};
constexpr BorderMode borders[] = {{Border::kConstant, cv::BORDER_CONSTANT},
                                  {Border::kReplicate, cv::BORDER_REPLICATE},
                                  {Border::kReflect, cv::BORDER_REFLECT},
                                  {Border::kReflect101, cv::BORDER_REFLECT_101},
                                  {Border::kWrap, cv::BORDER_WRAP},
                                  {Border::kTransparent, cv::BORDER_TRANSPARENT}};

cv::Mat cpu_warp(const cv::Mat& src, const cv::Mat& initial, const AffineMatrix& matrix,
                 int mode, int border, const BorderValue& value, bool inverse) {
  // NEAREST and LINEAR use the fixed CPU vector body. The native-width
  // scalar tail can round coordinates differently at sampling ties.
  const int width = mode == cv::INTER_CUBIC ? initial.cols :
      ((initial.cols + 255) / 128) * 128;
  cv::Mat wide(initial.rows, width, initial.type(), cv::Scalar::all(0));
  const cv::Rect rect(0, 0, initial.cols, initial.rows);
  initial.copyTo(wide(rect));
  const cv::Mat transform(2, 3, CV_64F, const_cast<double*>(matrix.data()));
  cv::Mat source = src;
  if (mode == cv::INTER_CUBIC && border == cv::BORDER_REFLECT_101 &&
      (src.cols == 1 || src.rows == 1)) {
    // The pinned CPU cubic helper loops for REFLECT_101 at length one.
    // All reflected taps on that axis equal the single input pixel.
    cv::repeat(src, src.rows == 1 ? 2 : 1, src.cols == 1 ? 2 : 1, source);
  }
  cv::warpAffine(source, wide, transform, wide.size(), mode | (inverse ? cv::WARP_INVERSE_MAP : 0),
                 border, cv::Scalar(value[0], value[1], value[2], value[3]));
  return wide(rect);
}

void compare_pixels(const cv::Mat& got, const cv::Mat& ref) {
  if (ref.depth() != CV_32F) {
    if (!bytes_equal(got, ref))
      throw std::runtime_error("integer pixels differ, max abs " +
                               std::to_string(cv::norm(got, ref, cv::NORM_INF)));
    return;
  }
  for (int y = 0; y < ref.rows; ++y) {
    for (int i = 0; i < ref.cols * ref.channels(); ++i) {
      const double expected = ref.ptr<float>(y)[i];
      const double error = std::abs(got.ptr<float>(y)[i] - expected);
      const double bound = std::min(1e-4, 5e-5 * std::max(std::abs(expected), 1.0));
      require(error <= bound, "warpAffine float error " + std::to_string(error));
    }
  }
}

void run_case(flagcv::Handle& handle, const Stream& stream, int type, cv::Size source_size,
              cv::Size destination_size, const AffineMatrix& matrix, Mode mode,
              BorderMode border, const BorderValue& value, bool inverse) {
  Image src(source_size.height, source_size.width, type, 3, 2, 5);
  Image dst(destination_size.height, destination_size.width, type, 5, 3, 2);
  cv::RNG rng(820);
  const int depth = CV_MAT_DEPTH(type);
  const double upper = depth == CV_8U ? 256 : (depth == CV_16U ? 65536 : 2);
  rng.fill(src.host, cv::RNG::UNIFORM, depth == CV_32F ? -2 : 0, upper);
  rng.fill(dst.host, cv::RNG::UNIFORM, depth == CV_32F ? -2 : 0, upper);
  const cv::Mat reference = cpu_warp(src.pixels(), dst.pixels(), matrix, mode.opencv,
                                    border.opencv, value, inverse);
  src.upload(stream.value);
  dst.upload(stream.value);
  flagcv::warpAffine(handle, src.view(), dst.view(), matrix, mode.flagcv,
                     border.flagcv, value, inverse, stream.value);
  const cv::Mat got = dst.download(stream);
  require(bytes_equal(src.host, src.download(stream)), "warpAffine changed its input");
  compare_pixels(got(dst.roi), reference);
  cv::Mat checked = got.clone();
  dst.pixels().copyTo(checked(dst.roi));
  require(bytes_equal(checked, dst.host), "warpAffine wrote outside its destination ROI");
}

void check_cubic_constant_fringe(flagcv::Handle& handle, const Stream& stream) {
  // With a large F32 constant, the CPU's weighted fringe differs from
  // writing the border value directly by more than the tolerance.
  const Mode cubic{Interpolation::kCubic, cv::INTER_CUBIC};
  const BorderMode constant{Border::kConstant, cv::BORDER_CONSTANT};
  for (const AffineMatrix& matrix : {AffineMatrix{0, 0, -2.2, 0, 0, 0},
                                     AffineMatrix{0, 0, 0, 0, 0, -2.2}}) {
    run_case(handle, stream, CV_32FC1, {4, 4}, {37, 1}, matrix, cubic, constant, {10000}, true);
  }
  for (const AffineMatrix& matrix : {AffineMatrix{1, 0, -3.2, 0, 1, 0},
                                     AffineMatrix{1, 0, 0, 1, 1, -3.2}}) {
    run_case(handle, stream, CV_32FC1, {4, 4}, {37, 1}, matrix, cubic, constant, {10000}, true);
  }
  for (int width : {5, 65, 133}) {
    const AffineMatrix tail{1, 0, -(width - 1) - 2.2, 0, 1, 0};
    run_case(handle, stream, CV_32FC1, {4, 4}, {width, 1}, tail, cubic, constant, {10000}, true);
  }
  for (int width : {1, 3, 129, 130, 131}) {
    const AffineMatrix padded{0, 0, -3.2, 0, 1, 0};
    run_case(handle, stream, CV_32FC1, {4, 4}, {width, 1}, padded, cubic, constant, {10000}, true);
  }
}

void check_integer_border_rounding(flagcv::Handle& handle, const Stream& stream) {
  constexpr double int32_max = std::numeric_limits<std::int32_t>::max();
  constexpr BorderMode constant{Border::kConstant, cv::BORDER_CONSTANT};
  constexpr AffineMatrix outside{0, 0, -128, 0, 0, -128};
  constexpr AffineMatrix partial{1, 0, -0.5, 0, 1, -0.5};
  constexpr BorderValue overflow{int32_max + 0.5, int32_max + 1, 1e12, -1e12};
  constexpr BorderValue ties{int32_max + 0.25, 2.5, 3.5, 65535.5};
  for (int type : {CV_8UC4, CV_16UC4}) {
    for (Mode mode : modes) {
      run_case(handle, stream, type, {4, 4}, {37, 2}, outside, mode, constant, overflow, true);
      run_case(handle, stream, type, {4, 4}, {37, 2}, partial, mode, constant, ties, true);
    }
  }
}

template<class Call>
void rejects(Call call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("warpAffine accepted an invalid request");
}

void reject_cases(flagcv::Handle& handle) {
  const Image src(2, 3, CV_8UC1), dst(2, 5, CV_8UC1), two(2, 5, CV_8UC2);
  const AffineMatrix identity{1, 0, 0, 0, 1, 0};
  rejects([&] { flagcv::warpAffine(handle, src.view(), src.view(), identity); });
  rejects([&] { flagcv::warpAffine(handle, src.view(), two.view(), identity); });
  rejects([&] {
    flagcv::warpAffine(handle, src.view(), dst.view(), identity,
                       Interpolation::kCubic, Border::kTransparent);
  });
  rejects([&] {
    flagcv::warpAffine(handle, src.view(), dst.view(), identity, static_cast<Interpolation>(-1));
  });
  rejects([&] {
    flagcv::warpAffine(handle, src.view(), dst.view(), identity, Interpolation::kNearest,
                       static_cast<Border>(-1));
  });
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { flagcv::warpAffine(handle, src.view(), dst.view(), {1, 0, nan, 0, 1, 0}); });
  rejects([&] {
    flagcv::warpAffine(handle, src.view(), dst.view(), {1, 0, 2147483392.0, 0, 1, 0},
                       Interpolation::kNearest, Border::kReflect, {}, true);
  });
}

void check_wide_indices(flagcv::Handle& handle) {
  const int width = 600000000;
  flagcv_test::DeviceMemory src(width);
  const flagcv::GpuMat source(src.ptr, width, 1, flagcv::Depth::k8U, 1);
  Image dst(1, 1, CV_8UC1);
  // REFLECT folds 1,100,000,000 to 99,999,999, including the repeated edge.
  // A second remainder after adding the period overflows signed i32.
  const unsigned char pixel = 173;
  cuda_check(cudaMemcpy(static_cast<unsigned char*>(src.ptr) + 99999999, &pixel,
                        sizeof(pixel), cudaMemcpyHostToDevice));
  const Stream stream(false);
  flagcv::warpAffine(handle, source, dst.view(), {1, 0, 1100000000, 0, 1, 0},
                     Interpolation::kNearest, Border::kReflect, {}, true);
  require(dst.download(stream).at<unsigned char>(0, 0) == 173, "large reflect index");

  Image narrow(1, 1, CV_8UC1), wide(1, 128 * 65535 + 1, CV_8UC1);
  narrow.host.setTo(91);
  wide.host.setTo(23);
  narrow.upload(stream.value);
  wide.upload(stream.value);
  flagcv::warpAffine(handle, narrow.view(), wide.view(), {1, 0, 0, 0, 1, 0},
                     Interpolation::kNearest, Border::kReplicate);
  const cv::Mat got = wide.download(stream);
  require(cv::countNonZero(got != 91) == 0, "wide output grid split");
}

}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef FLAGCV_TEST_HIP
    flagcv::Handle handle("hip");
#else
    flagcv::Handle handle;
#endif
    cv::setNumThreads(1);
    const Stream default_stream(false), native_stream;
    check_integer_border_rounding(handle, native_stream);
    if (argc == 2 && std::string(argv[1]) == "--integer-border") {
      std::cout << "warpAffine integer border: 16 cases passed\n";
      return 0;
    }
    reject_cases(handle);
    check_wide_indices(handle);
    check_cubic_constant_fringe(handle, native_stream);
    int cases = 0;
    int failures = 0;
    for (int depth : {CV_8U, CV_16U, CV_32F}) {
      for (int channels : {1, 3, 4}) {
        for (const Mode mode : modes) {
          for (const BorderMode border : borders) {
            if (mode.flagcv == Interpolation::kCubic && border.flagcv == Border::kTransparent)
              continue;
            for (bool inverse : {false, true}) {
              const int id = cases++;
              try {
                const auto& stream = inverse ? native_stream : default_stream;
                run_case(handle, stream, CV_MAKETYPE(depth, channels), {53, 37}, {37, 53},
                         {1, 1.9, -20, 0.4, 1, -10}, mode, border,
                         {2.5, 123, -0.75, 9}, inverse);
                run_case(handle, stream, CV_MAKETYPE(depth, channels), {1, 7}, {9, 7},
                         {1, 0, 0.5, 0, 1, -0.5}, mode, border,
                         {3.5, 4.25, -2.5, 7}, inverse);
                std::cout << "PASS " << id;
              } catch (const std::exception& error) {
                ++failures;
                std::cout << "FAIL " << id << ' ' << error.what();
              }
              std::cout << " depth=" << depth << " channels=" << channels
                        << " mode=" << mode.opencv << " border=" << border.opencv
                        << " inverse=" << inverse << '\n';
            }
          }
        }
      }
    }
    std::cout << "warpAffine " << cases - failures << '/' << cases << " cases passed\n";
    return failures != 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
