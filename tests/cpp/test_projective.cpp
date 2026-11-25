// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <limits>

#include "cuda_image.hpp"

namespace {
using flagcv::Border;
using flagcv::BorderValue;
using flagcv::Interpolation;
using flagcv::PerspectiveMatrix;
using flagcv_test::Image;
using flagcv_test::Stream;
using flagcv_test::bytes_equal;
using flagcv_test::require;

constexpr struct { Interpolation gpu; int cpu; } modes[] = {
    {Interpolation::kNearest, cv::INTER_NEAREST}, {Interpolation::kLinear, cv::INTER_LINEAR},
    {Interpolation::kCubic, cv::INTER_CUBIC}, {Interpolation::kArea, cv::INTER_AREA}};
constexpr struct { Border gpu; int cpu; } borders[] = {
    {Border::kConstant, cv::BORDER_CONSTANT}, {Border::kReplicate, cv::BORDER_REPLICATE},
    {Border::kReflect, cv::BORDER_REFLECT}, {Border::kWrap, cv::BORDER_WRAP},
    {Border::kReflect101, cv::BORDER_REFLECT_101}, {Border::kTransparent, cv::BORDER_TRANSPARENT}};
constexpr BorderValue value{2.5, -1.25, 123, 9};

void check(const Image& src, const Image& dst, const Stream& stream, const cv::Mat& ref) {
  cv::Mat host = dst.download(stream);
  const auto got = host(dst.roi);
  for (int y = 0; y < got.rows; ++y) {
    for (int x = 0; x < got.cols * got.channels(); ++x) {
      double actual, expected;
      if (got.depth() == CV_8U) {
        actual = got.ptr<std::uint8_t>(y)[x];
        expected = ref.ptr<std::uint8_t>(y)[x];
      } else if (got.depth() == CV_16U) {
        actual = got.ptr<std::uint16_t>(y)[x];
        expected = ref.ptr<std::uint16_t>(y)[x];
      } else {
        actual = got.ptr<float>(y)[x];
        expected = ref.ptr<float>(y)[x];
      }
      if (actual == expected || (std::isnan(actual) && std::isnan(expected))) continue;
      const double bound = got.depth() == CV_32F && std::isfinite(expected)
                               ? std::min(1e-4, 5e-5 * std::max(1.0, std::abs(expected))) : 0;
      require(std::abs(actual - expected) <= bound,
              "pixel (" + std::to_string(x) + "," + std::to_string(y) + ") got " +
                  std::to_string(actual) + " expected " + std::to_string(expected));
    }
  }
  dst.pixels().copyTo(host(dst.roi));
  require(bytes_equal(host, dst.host), "geometry modified destination guards");
  require(bytes_equal(src.host, src.download(stream)), "geometry modified source");
}

void fill(Image& src, Image& dst) {
  const int depth = src.host.depth();
  const double high = depth == CV_8U ? 256 : (depth == CV_16U ? 65536 : 2);
  cv::RNG rng(18458);
  for (Image* image : {&src, &dst})
    rng.fill(image->host, cv::RNG::UNIFORM, depth == CV_32F ? -2 : 0, high);
}

void perspective(flagcv::Handle& handle, const Stream& stream, int type, int interp, int border,
                 bool inverse, const PerspectiveMatrix& matrix, int cols = 137, int rows = 13) {
  Image src(17, 31, type, 3, 2, 5), dst(rows, cols, type, 5, 3, 2);
  fill(src, dst);
  cv::Mat ref = dst.pixels().clone();
  const cv::Mat transform(3, 3, CV_64F, const_cast<double*>(matrix.data()));
  cv::warpPerspective(src.pixels(), ref, transform, ref.size(),
                      modes[interp].cpu | (inverse ? cv::WARP_INVERSE_MAP : 0),
                      borders[border].cpu, cv::Scalar(value[0], value[1], value[2], value[3]),
                      cv::ALGO_HINT_ACCURATE);
  src.upload(stream.value);
  dst.upload(stream.value);
  flagcv::warpPerspective(handle, src.view(), dst.view(), matrix, modes[interp].gpu,
                          borders[border].gpu, value, inverse, stream.value);
  check(src, dst, stream, ref);
}

void remapping(flagcv::Handle& handle, const Stream& stream, int type, int interp, int border,
               int layout, bool relative, int cols = 131) {
  Image src(17, 31, type, 3, 2, 5), dst(11, cols, type, 5, 3, 2);
  const int map_type = layout == 0 ? CV_32FC2 : (layout == 1 ? CV_32FC1 : CV_16SC2);
  Image map1(11, cols, map_type, 2, 4, 3);
  Image map2(11, cols, layout == 1 ? CV_32FC1 : (layout == 3 ? CV_16SC1 : CV_16UC1), 3, 1, 4);
  const bool second = layout != 0 && layout != 4;
  fill(src, dst);
  map1.host.setTo(cv::Scalar::all(12));
  map2.host.setTo(cv::Scalar::all(6));
  cv::Mat coordinates(11, cols, CV_32FC2);
  for (int y = 0; y < coordinates.rows; ++y) {
    for (int x = 0; x < coordinates.cols; ++x) {
      const float u = float((x * 23 + y * 9) % 337) / 10 - 1.11f;
      const float v = float((x * 5 + y * 63) % 191) / 10 - 1.11f;
      coordinates.at<cv::Vec2f>(y, x) = {u - (relative ? x : 0), v - (relative ? y : 0)};
    }
  }
  cv::Mat cpu1 = map1.pixels(), cpu2 = second ? map2.pixels() : cv::Mat();
  if (layout == 0) {
    coordinates.copyTo(cpu1);
  } else if (layout == 1) {
    cv::extractChannel(coordinates, cpu1, 0);
    cv::extractChannel(coordinates, cpu2, 1);
  } else {
    cv::Mat fraction;
    cv::convertMaps(coordinates, cv::noArray(), cpu1, fraction, CV_16SC2, !second);
    if (second) {
      if (layout == 3) cpu2 = cv::Mat(fraction.size(), CV_16SC1, fraction.data);
      else fraction.copyTo(cpu2);
      if (layout == 3) cpu2.copyTo(map2.pixels());
      cpu2 = map2.pixels();
    }
  }
  cv::Mat ref = dst.pixels().clone();
  cv::remap(src.pixels(), ref, cpu1, cpu2,
            modes[interp].cpu | (relative ? cv::WARP_RELATIVE_MAP : 0),
            borders[border].cpu, cv::Scalar(value[0], value[1], value[2], value[3]),
            cv::ALGO_HINT_ACCURATE);
  src.upload(stream.value);
  dst.upload(stream.value);
  map1.upload(stream.value);
  if (second) map2.upload(stream.value);
  flagcv::remap(handle, src.view(), dst.view(), map1.view(),
                second ? map2.view() : flagcv::GpuMat{}, modes[interp].gpu,
                borders[border].gpu, value, relative, stream.value);
  check(src, dst, stream, ref);
  require(bytes_equal(map1.host, map1.download(stream)), "geometry modified coordinates");
  if (second) require(bytes_equal(map2.host, map2.download(stream)), "geometry modified fractions");
}
void cubic_nonfinite(flagcv::Handle& handle, const Stream& stream, int border,
                     float pixel, float coordinate, int width, bool map) {
  Image src(1, 1, CV_32FC1), dst(1, width, CV_32FC1), xy(1, width, CV_32FC2);
  src.host.setTo(pixel);
  dst.host.setTo(13);
  xy.host.setTo(cv::Scalar(coordinate, 0));
  auto ref = dst.pixels().clone();
  src.upload(stream.value);
  dst.upload(stream.value);
  const cv::Scalar constant(7);
  const BorderValue bv{7};
  if (map) {
    cv::remap(src.pixels(), ref, xy.pixels(), cv::noArray(), cv::INTER_CUBIC,
              borders[border].cpu, constant);
    xy.upload(stream.value);
    flagcv::remap(handle, src.view(), dst.view(), xy.view(), {}, Interpolation::kCubic,
                  borders[border].gpu, bv, false, stream.value);
  } else {
    const PerspectiveMatrix matrix{0, 0, coordinate, 0, 0, 0, 0, 0, 1};
    const cv::Mat transform(3, 3, CV_64F, const_cast<double*>(matrix.data()));
    cv::warpPerspective(src.pixels(), ref, transform, ref.size(),
                        cv::INTER_CUBIC | cv::WARP_INVERSE_MAP, borders[border].cpu, constant);
    flagcv::warpPerspective(handle, src.view(), dst.view(), matrix, Interpolation::kCubic,
                             borders[border].gpu, bv, true, stream.value);
  }
  check(src, dst, stream, ref);
}

}  // namespace

int main(int argc, char** argv) {
  cv::setNumThreads(1);
  const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
  const bool fp32_edges = argc == 2 && std::string(argv[1]) == "--fp32-cubic-edges";
  const auto backend =
#ifdef FLAGCV_TEST_HIP
      "hip";
#else
      "nvidia";
#endif
  flagcv::Handle handle(backend, 0);
  const Stream custom, standard(false);
  const PerspectiveMatrix general{1.08, 0.043, -1.49, -0.02, 0.93, -0.51, 0.002, -0.003, 1};
  const PerspectiveMatrix pole{0.3, 0, -1, 0.1, 0.3, 0, 1, 0, -4};
  const PerspectiveMatrix sub_float_pole{0.3, 0, -1, 0.1, 0.3, 0, 1, 0, -4.000000000001};
  const PerspectiveMatrix near_pole{0.3, 0, -1, 0.1, 0.3, 0, 1, 0, -4.000000476837158203125};
  int total = 0, failed = 0;
  auto test = [&](const std::string& label, auto run) {
    ++total;
    try {
      run();
      std::cout << "PASS " << label << std::endl;
    } catch (const std::exception& error) {
      ++failed;
      std::cout << "FAIL " << label << ' ' << error.what() << std::endl;
    }
  };
  for (int depth : {CV_8U, CV_16U, CV_32F}) {
    if (fp32_edges && depth != CV_32F) continue;
    for (int cn : {1, 3, 4}) {
      if (smoke && cn != 1) continue;
      const auto type = CV_MAKETYPE(depth, cn);
      for (int interp = 0; interp < (smoke ? 3 : 4); ++interp) {
        if (fp32_edges && interp != 2) continue;
        for (int border = 0; border < 6; ++border) {
          if ((smoke || fp32_edges) && border != 0 && border != 5) continue;
          const auto bits = " type=" + std::to_string(type) + " interpolation=" +
                            std::to_string(interp) + " border=" + std::to_string(border);
          for (bool inverse : {false, true}) {
            if (smoke && !inverse) continue;
            test("perspective" + bits + " inverse=" + std::to_string(inverse), [&] {
              perspective(handle, inverse ? custom : standard, type, interp, border, inverse, general);
            });
          }
          for (int layout = 0; layout < (interp == 0 ? 5 : 4); ++layout) {
            if (smoke && layout != 0 && layout != 2) continue;
            for (bool relative : {false, true}) {
              if (smoke && relative) continue;
              test("remap" + bits + " layout=" + std::to_string(layout) +
                       " relative=" + std::to_string(relative), [&] {
                remapping(handle, relative ? custom : standard, type, interp, border, layout, relative);
              });
            }
          }
        }
      }
    }
  }
  if (!smoke && !fp32_edges) {
    for (int interp = 0; interp < 3; ++interp) {
      test("perspective width 32768 interpolation=" + std::to_string(interp), [&] {
        perspective(handle, custom, CV_8UC1, interp, 1, true,
                    {1, 0, 0, 0, 1, 0, 0, 0, 1}, 32768, 1);
      });
    }
    test("perspective width crossing the second grid axis", [&] {
      perspective(handle, custom, CV_8UC1, 0, 1, true,
                  {1, 0, 0, 0, 1, 0, 0, 0, 1}, 8388481, 1);
    });
  }
  if (!smoke && !fp32_edges) {
    test("invalid map, output and matrix contracts", [&] {
      Image src(5, 7, CV_8UC1), dst(5, 7, CV_8UC1), f1(5, 7, CV_32FC1),
            f2(5, 7, CV_32FC2), fixed(5, 7, CV_16SC2), fraction(5, 7, CV_16UC1),
            wrong(4, 7, CV_32FC2), wrong_depth(5, 7, CV_16UC1);
      auto rejects = [](auto call) {
        try {
          call();
        } catch (const std::invalid_argument&) {
          return;
        }
        throw std::runtime_error("invalid geometry call was accepted");
      };
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), {}); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), f1.view()); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), wrong.view()); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), f2.view(), f1.view()); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), fixed.view()); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), fraction.view(), fixed.view(),
                                  Interpolation::kCubic); });
      rejects([&] { flagcv::remap(handle, src.view(), src.view(), f2.view()); });
      rejects([&] { flagcv::warpPerspective(handle, src.view(), wrong_depth.view(), general); });
      rejects([&] { flagcv::warpPerspective(handle, src.view(), src.view(), general); });
      auto bad = general;
      bad[8] = std::numeric_limits<double>::quiet_NaN();
      rejects([&] { flagcv::warpPerspective(handle, src.view(), dst.view(), bad); });
      rejects([&] { flagcv::warpPerspective(handle, src.view(), dst.view(), general,
                                            static_cast<Interpolation>(-1)); });
      rejects([&] { flagcv::remap(handle, src.view(), dst.view(), f2.view(), {},
                                  Interpolation::kLinear, static_cast<Border>(-1)); });
      auto alias = flagcv::GpuMat(f1.view().data(), 7, 5, flagcv::Depth::k32F, 1);
      Image float_src(5, 7, CV_32FC1);
      rejects([&] { flagcv::remap(handle, float_src.view(), alias, f1.view(), f1.view()); });
    });
  }
  for (int border : {0, 5})
    for (float pixel : {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()})
      for (bool map : {false, true})
        for (int width : {1, 4})
          test("cubic nonfinite border=" + std::to_string(border) + " map=" +
                   std::to_string(map) + " width=" + std::to_string(width) +
                   " pixel=" + std::to_string(pixel), [&] {
            cubic_nonfinite(handle, custom, border, pixel, width == 1 ? -2.5f : -3.5f,
                            width, map);
          });
  for (int interp = 0; interp < 3; ++interp)
    for (int border : {0, 1, 3, 5})
      for (const auto& matrix : {pole, near_pole, sub_float_pole, PerspectiveMatrix{}}) {
        if (fp32_edges) continue;
        test("perspective pole interpolation=" + std::to_string(interp) +
                 " border=" + std::to_string(border) + " offset=" +
                 std::to_string((matrix[8] + 4) * 1e12), [&] {
          perspective(handle, custom, CV_16UC1, interp, border, true, matrix, 17);
        });
      }
  std::cout << "projective " << total - failed << '/' << total << " cases passed\n";
  return failed != 0;
}
