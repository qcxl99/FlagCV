// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include "cuda_image.hpp"
#include "resize_reference.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

namespace {
using flagcv::DeviceWorkspace;
using flagcv::Interpolation;
using flagcv::ResizeOptions;
using flagcv::ResizePlan;
using flagcv::Size;
using flagcv_test::bytes_equal;
using flagcv_test::cuda_check;
using flagcv_test::DeviceMemory;
using flagcv_test::Image;
using flagcv_test::require;
using flagcv_test::Stream;

constexpr Interpolation modes[] = {Interpolation::kNearest, Interpolation::kLinear,
                                   Interpolation::kCubic, Interpolation::kArea};

class Tables {
 public:
  Tables(const ResizePlan& plan, const Stream& stream) : bytes_(plan.table_bytes()) {
    if (bytes_) {
      data_ = std::make_unique<DeviceMemory>(bytes_);
      cuda_check(cudaMemcpyAsync(data_->ptr, plan.table_data(), bytes_,
                                 cudaMemcpyHostToDevice, stream.value));
    }
  }

  DeviceWorkspace view() const { return {data_ ? data_->ptr : nullptr, bytes_}; }

 private:
  std::size_t bytes_;
  std::unique_ptr<DeviceMemory> data_;
};

cv::Mat reference(const cv::Mat& src, Size size, const ResizeOptions& options) {
  cv::Mat dst(size.height, size.width, src.type());
  const bool explicit_size = options.size.width > 0 && options.size.height > 0;
  const double fx = explicit_size ? double(size.width) / src.cols : options.fx;
  const double fy = explicit_size ? double(size.height) / src.rows : options.fy;
  using flagcv::resize_tables::ImageRef;
  const ImageRef input{src.data, src.rows, src.cols, src.channels(), src.step[0], src.depth()==CV_32F};
  const ImageRef output{dst.data, dst.rows, dst.cols, dst.channels(), dst.step[0], dst.depth()==CV_32F};
  flagcv::resize_tables::resize(input, output, fy, fx, static_cast<int>(options.interpolation));
  return dst;
}

void compare(const cv::Mat& actual, const cv::Mat& expected, Interpolation mode) {
  if (expected.depth() == CV_8U || mode == Interpolation::kNearest) {
    require(bytes_equal(actual, expected), "resize pixel bits differ");
  } else {
    const double bound = mode == Interpolation::kLinear ? 4.948e-7 :
        mode == Interpolation::kCubic ? 1.401e-6 : 8.419e-7;
    const auto error = cv::norm(actual, expected, cv::NORM_INF);
    require(error <= bound, "resize FP32 delta " + std::to_string(error));
  }
}

void compare_roi(const Image& image, cv::Mat actual, const cv::Mat& expected, Interpolation mode) {
  compare(actual(image.roi), expected, mode);
  image.pixels().copyTo(actual(image.roi));
  require(bytes_equal(actual, image.host), "resize wrote outside its destination ROI");
}

void run_case(flagcv::Handle& handle, const Stream& native, const Stream& second,
              Size in, ResizeOptions options, int type) {
  const ResizePlan plan(in, flagcv_test::depth_for(CV_MAT_DEPTH(type)),
                        CV_MAT_CN(type), options);
  const Size out = plan.destination_size();
  Image src(in.height, in.width, type, 3, 2, 5);
  Image one(out.height, out.width, type, 3, 1, 7);
  Image two(out.height, out.width, type, 1, 2, 5);
  cv::RNG rng(820);
  rng.fill(src.host, cv::RNG::UNIFORM, src.host.depth() == CV_8U ? 0 : -2,
           src.host.depth() == CV_8U ? 256 : 2);
  one.host.setTo(cv::Scalar::all(23));
  two.host.setTo(cv::Scalar::all(71));
  const cv::Mat oracle = reference(src.pixels(), out, options);

  Tables tables(plan, native);
  src.upload(native.value);
  one.upload(native.value);
  two.upload(second.value);
  native.wait();  // Both streams may use the same immutable uploaded coefficients.
  flagcv::resize(handle, src.view(), one.view(), plan, tables.view(), native.value);
  flagcv::resize(handle, src.view(), two.view(), plan, tables.view(), second.value);
  compare_roi(one, one.download(native), oracle, options.interpolation);
  compare_roi(two, two.download(second), oracle, options.interpolation);
  require(bytes_equal(src.host, src.download(native)), "resize changed its source bytes");
}

flagcv::GpuMat subview(const Image& backing, const cv::Rect& roi) {
  const auto data = static_cast<unsigned char*>(backing.device.ptr) +
      roi.y * backing.host.step + roi.x * backing.host.elemSize();
  return {data, roi.width, roi.height, flagcv_test::depth_for(backing.host.depth()),
          backing.host.channels(), static_cast<std::int64_t>(backing.host.step)};
}

void run_batch(flagcv::Handle& handle, const Stream& stream, Size in, ResizeOptions options,
               int channels) {
  constexpr int count = 3;
  const ResizePlan plan(in, flagcv::Depth::k8U, channels, options);
  const Size out = plan.destination_size();
  const int source_rows = in.height + 3;
  const int destination_rows = out.height + 5;
  const int type = CV_MAKETYPE(CV_8U, channels);
  Image source(count * source_rows, in.width + 8, type);
  Image dest(count * destination_rows, out.width + 7, type);
  cv::RNG rng(912);
  rng.fill(source.host, cv::RNG::UNIFORM, 0, 256);
  rng.fill(dest.host, cv::RNG::UNIFORM, 0, 256);
  cv::Rect src_roi(2, 1, in.width, in.height), dst_roi(3, 2, out.width, out.height);
  Tables tables(plan, stream);
  source.upload(stream.value);
  dest.upload(stream.value);
  flagcv::resizeBatch(handle, subview(source, src_roi), subview(dest, dst_roi), count,
                      source.host.step * source_rows, dest.host.step * destination_rows,
                      plan, tables.view(), stream.value);
  auto got = dest.download(stream);
  for (int n = 0; n < count; ++n) {
    compare(got(dst_roi), reference(source.host(src_roi), out, options), options.interpolation);
    dest.host(dst_roi).copyTo(got(dst_roi));
    src_roi.y += source_rows;
    dst_roi.y += destination_rows;
  }
  require(bytes_equal(got, dest.host), "resizeBatch wrote outside the destination views");
  require(bytes_equal(source.host, source.download(stream)), "resizeBatch changed its source");
}

template<class Call>
void rejects(Call call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("resize accepted an invalid argument");
}

void reject_cases(flagcv::Handle& handle, const Stream& stream) {
  Image src(4, 4, CV_8UC1), dst(6, 6, CV_8UC1);
  const ResizePlan plan({4, 4}, flagcv::Depth::k8U, 1, {{6, 6}, 0, 0});
  const ResizePlan copy({4, 4}, flagcv::Depth::k8U, 1, {{4, 4}, 0, 0});
  const ResizePlan partial({4, 4}, flagcv::Depth::k8U, 1, {{0, 6}, 1.5, 1.5});
  require(partial.destination_size().width == 6, "resize partial size precedence");
  rejects([&] { flagcv::resize(handle, src.view(), dst.view(), plan); });
  rejects([&] { flagcv::resize(handle, src.view(), src.view(), copy); });
  rejects([&] { flagcv::resizeBatch(handle, src.view(), dst.view(), 1, 0, 0, partial); });
  Tables tables(plan, stream);
  const auto workspace = tables.view();
  rejects([&] {
    flagcv::resize(handle, src.view(), dst.view(), plan, {workspace.data, workspace.bytes - 1});
  });
  rejects([&] {
    flagcv::resize(handle, src.view(), dst.view(), plan,
                   {static_cast<const char*>(workspace.data) + 1, workspace.bytes});
  });
  rejects([&] {
    flagcv::resize(handle, src.view(), dst.view(), plan, {dst.view().data(), workspace.bytes});
  });
  Image backing(16, 4, CV_8UC1);
  rejects([&] {
    flagcv::resizeBatch(handle, subview(backing, {0, 0, 4, 4}),
                        subview(backing, {0, 4, 4, 4}), 2, 16, 16, copy);
  });
}

void check_wide_area_sum(flagcv::Handle& handle, const Stream& stream) {
  Image src(2160, 4096, CV_8UC1), dst(1, 1, CV_8UC1);
  src.host.setTo(255);
  src.upload(stream.value);
  const ResizePlan plan({4096, 2160}, flagcv::Depth::k8U, 1, {{1, 1}, 0, 0, Interpolation::kArea});
  require(plan.table_bytes() == 0, "integral AREA should have no tables");
  flagcv::resize(handle, src.view(), dst.view(), plan, {}, stream.value);
  require(dst.download(stream).at<std::uint8_t>(0, 0) == 255, "wide AREA sum overflowed");
}

}  // namespace

int main() {
  try {
#ifdef FLAGCV_TEST_HIP
    flagcv::Handle handle("hip");
#else
    flagcv::Handle handle;
#endif
    const Stream native, default_stream(false);
    reject_cases(handle, native);
    for (auto mode : modes) {
      for (int channels : {1, 3, 4}) {
        for (int depth : {CV_8U, CV_32F}) {
          const int type = CV_MAKETYPE(depth, channels);
          run_case(handle, native, default_stream, {6, 17}, {{34, 37}, 0, 0, mode}, type);
          run_case(handle, native, default_stream, {60, 32}, {{}, .5, .5, mode}, type);
          run_case(handle, native, default_stream, {37, 29}, {{17, 13}, 0, 0, mode}, type);
          run_case(handle, native, default_stream, {1, 11}, {{7, 4}, 0, 0, mode}, type);
          run_case(handle, native, default_stream, {9, 7}, {{9, 7}, 0, 0, mode}, type);
        }
        run_batch(handle, native, {19, 13}, {{}, .68, .68, mode}, channels);
        run_batch(handle, native, {12, 18}, {{6, 9}, 0, 0, mode}, channels);
      }
      std::cout << "resize mode " << static_cast<int>(mode) << " passed\n";
    }
    check_wide_area_sum(handle, native);
    std::cout << "resize 120 single, 24 batch and wide AREA cases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
