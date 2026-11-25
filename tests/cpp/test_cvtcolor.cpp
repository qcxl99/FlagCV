// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include "cuda_image.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>

namespace {
using flagcv::ColorTables;
using flagcv::DeviceWorkspace;
using flagcv::Depth;
using flagcv_test::bytes_equal;
using flagcv_test::cuda_check;
using flagcv_test::DeviceMemory;
using flagcv_test::Image;
using flagcv_test::require;
using flagcv_test::Stream;

struct Case {
  int code;
  int channels;
  int depth;
  int width_multiple;
  bool yuv420;
  bool bit_copy;
};

constexpr Case cases[] = {
#include "cvtcolor_cases.inc"
};
static_assert(std::size(cases) == 124);

class Tables {
 public:
  DeviceWorkspace on_device(ColorTables kind, const Stream& upload) {
    const auto index = static_cast<std::size_t>(kind);
    if (!handles_[index] && kind != ColorTables::kNone) {
      const auto host = flagcv::cvtColorTableData(kind);
      handles_[index] = std::make_unique<DeviceMemory>(host.bytes);
      size_[index] = host.bytes;
      cuda_check(cudaMemcpyAsync(handles_[index]->ptr, host.data, host.bytes,
                                 cudaMemcpyHostToDevice, upload.value));
      upload.wait();
    }
    return {handles_[index] ? handles_[index]->ptr : nullptr, size_[index]};
  }

 private:
  std::array<std::unique_ptr<DeviceMemory>, 5> handles_;
  std::array<std::size_t, 5> size_{};
};

void compare(const Case& entry, const cv::Mat& source, const cv::Mat& actual,
             const cv::Mat& expected) {
  if (entry.depth != CV_32F || entry.bit_copy) {
    require(bytes_equal(actual, expected), "cvtColor bits differ");
    return;
  }
  require(actual.size() == expected.size() && actual.type() == expected.type(),
          "cvtColor CPU layout differs");
  for (int y = 0; y < expected.rows; ++y) {
    const auto* input = source.ptr<float>(y);
    const auto* got = actual.ptr<float>(y);
    const auto* ref = expected.ptr<float>(y);
    for (int x = 0; x < expected.cols; ++x) {
      double input_abs = 0;
      for (int c = 0; c < source.channels(); ++c)
        input_abs = std::max(input_abs, std::abs(double(input[x * source.channels() + c])));
      for (int c = 0; c < expected.channels(); ++c) {
        const int index = x * expected.channels() + c;
        const double delta = std::abs(double(got[index]) - ref[index]);
        const double bound = 8.0 * std::ldexp(1.0, -24) *
                             std::max({1.0, input_abs, std::abs(double(ref[index]))});
        require(std::isfinite(got[index]) && std::isfinite(ref[index]) &&
                    delta <= bound && delta <= 1e-4,
                "cvtColor FP32 differs " + std::to_string(delta));
      }
    }
  }
}

void run_case(flagcv::Handle& handle, Tables& tables, const Stream& upload, const Stream& stream,
              const Case& entry, int width, bool large_float) {
  if (width % entry.width_multiple != 0) ++width;
  const int src_type = CV_MAKETYPE(entry.depth, entry.channels);
  const int rows = entry.yuv420 ? 15 : 11;
  Image src(rows, width, src_type, 1, 2, 5);
  if (entry.depth == CV_32F && entry.bit_copy) {
    // Copy cases preserve IEEE storage, including NaN payload, infinity and signed zero.
    constexpr std::uint32_t bits[] = {0x7fc00123u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x3f800000u};
    for (std::size_t i = 0; i < src.host.total() * src.host.channels(); ++i)
      std::memcpy(src.host.data + i * sizeof(float), &bits[i % std::size(bits)], sizeof(float));
  } else {
    const double upper = entry.depth == CV_8U ? 256 : entry.depth == CV_16U ? 65536 :
                         large_float ? 131070 : 2;
    cv::RNG random(143 + entry.code);
    random.fill(src.host, cv::RNG::UNIFORM, entry.depth == CV_32F ? -upper : 0, upper);
  }

  cv::Mat expected;
  cv::cvtColor(src.pixels(), expected, entry.code);
  const auto shape = flagcv::cvtColorOutput({width, rows}, flagcv_test::depth_for(entry.depth),
                                           entry.channels, entry.code);
  require(shape.size.width == expected.cols && shape.size.height == expected.rows &&
              shape.channels == expected.channels(),
          "cvtColor output query disagrees with direct CPU output");
  Image dst(expected.rows, expected.cols, expected.type(), 3, 1, 7);
  dst.host.setTo(cv::Scalar::all(19));
  const auto workspace = tables.on_device(flagcv::cvtColorTableKind(entry.code), upload);
  src.upload(stream.value);
  dst.upload(stream.value);
  flagcv::cvtColor(handle, src.view(), dst.view(), entry.code, workspace, stream.value);
  auto received = dst.download(stream);
  compare(entry, src.pixels(), received(dst.roi), expected);
  require(bytes_equal(src.host, src.download(stream)), "cvtColor changed the input");
  dst.pixels().copyTo(received(dst.roi));
  require(bytes_equal(received, dst.host), "cvtColor wrote outside the destination ROI");
}

void check_family_stream_reuse(flagcv::Handle& handle, Tables& tables,
                               const Stream& first, const Stream& second) {
  const int bgr = cv::COLOR_BGR2Luv, rgb = cv::COLOR_RGB2Luv;
  const auto family = flagcv::cvtColorTableKind(bgr);
  require(family == flagcv::cvtColorTableKind(rgb), "forward Luv payload should be shared");
  const auto workspace = tables.on_device(family, first);
  Image src(9, 17, CV_8UC3), a(9, 17, CV_8UC3), b(9, 17, CV_8UC3);
  cv::RNG random(7752);
  random.fill(src.host, cv::RNG::UNIFORM, 0, 256);
  cv::Mat expected_a, expected_b;
  cv::cvtColor(src.host, expected_a, bgr);
  cv::cvtColor(src.host, expected_b, rgb);
  src.upload(first.value);
  first.wait();
  for (int i = 0; i < 3; ++i) {
    flagcv::cvtColor(handle, src.view(), a.view(), bgr, workspace, first.value);
    flagcv::cvtColor(handle, src.view(), b.view(), rgb, workspace, second.value);
  }
  require(bytes_equal(a.download(first), expected_a) && bytes_equal(b.download(second), expected_b),
          "cvtColor shared immutable Luv table differs across streams");
}

template<class Call>
void rejects(Call call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("cvtColor accepted an invalid argument");
}

void reject_cases(flagcv::Handle& handle, Tables& tables, const Stream& stream) {
  Image src(11, 17, CV_8UC3), dst(11, 17, CV_8UC3);
  rejects([&] { flagcv::cvtColor(handle, src.view(), dst.view(), cv::COLOR_HSV2BGR); });
  rejects([&] { flagcv::cvtColor(handle, src.view(), src.view(), cv::COLOR_BGR2RGB); });
  rejects([&] { flagcv::cvtColor(handle, src.view(), dst.view(), cv::COLOR_BGR2GRAY); });
  rejects([&] { flagcv::cvtColor(handle, src.view(), dst.view(), cv::COLOR_BGR2HSV); });
  rejects([&] { flagcv::cvtColorOutput({17, 11}, Depth::k32F, 2, cv::COLOR_YUV2GRAY_YUY2); });
  rejects([&] { flagcv::cvtColorOutput({17, 11}, Depth::k8U, 2, cv::COLOR_YUV2BGR_YUY2); });
  rejects([&] { flagcv::cvtColorOutput({18, 11}, Depth::k8U, 1, cv::COLOR_YUV2BGR_NV12); });

  const auto workspace = tables.on_device(ColorTables::kLabLuvReverse, stream);
  rejects([&] {
    flagcv::cvtColor(handle, src.view(), dst.view(), cv::COLOR_Luv2BGR,
                    {static_cast<const char*>(workspace.data) + 4, workspace.bytes});
  });
  rejects([&] {
    flagcv::cvtColor(handle, src.view(), dst.view(), cv::COLOR_BGR2Lab,
                     {workspace.data, workspace.bytes - 1});
  });
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
    Tables tables;
    reject_cases(handle, tables, native);
    int errors = 0;
    for (const auto& entry : cases) {
      try {
        run_case(handle, tables, native, native, entry, 17, false);
        run_case(handle, tables, native, default_stream, entry, 5, true);
        std::cout << "PASS code=" << entry.code << " depth=" << entry.depth << '\n';
      } catch (const std::exception& e) {
        ++errors;
        std::cout << "FAIL code=" << entry.code << " depth=" << entry.depth << " " << e.what() << '\n';
      }
    }
    check_family_stream_reuse(handle, tables, native, default_stream);
    std::cout << "cvtColor " << std::size(cases) - errors << "/" << std::size(cases)
              << " code-depth combinations passed\n";
    return errors ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
