// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <limits>
#include <stdexcept>

#include "cuda_image.hpp"

namespace {
using flagcv::Border;
using flagcv::Depth;
using flagcv::DeviceWorkspace;
using flagcv::Filter2DOptions;
using flagcv::Filter2DPlan;
using flagcv::GaussianOptions;
using flagcv::GaussianPlan;
using flagcv::Handle;
using flagcv::Size;
using flagcv_test::bytes_equal;
using flagcv_test::cuda_check;
using flagcv_test::depth_for;
using flagcv_test::DeviceMemory;
using flagcv_test::Image;
using flagcv_test::require;
using flagcv_test::Stream;

struct BorderPair { Border flag; int cv; };
constexpr BorderPair borders[] = {
    {Border::kConstant, cv::BORDER_CONSTANT},
    {Border::kReplicate, cv::BORDER_REPLICATE},
    {Border::kReflect, cv::BORDER_REFLECT},
    {Border::kReflect101, cv::BORDER_REFLECT_101}};

class Table {
 public:
  Table(flagcv::HostTableData host, const Stream& upload)
      : memory_(host.bytes), bytes_(host.bytes) {
    cuda_check(cudaMemcpyAsync(memory_.ptr, host.data, host.bytes,
                               cudaMemcpyHostToDevice, upload.value));
  }
  DeviceWorkspace view() const { return {memory_.ptr, bytes_}; }
 private:
  DeviceMemory memory_;
  std::size_t bytes_;
};

void compare(const cv::Mat& actual, const cv::Mat& expected) {
  if (expected.depth() != CV_32F && expected.depth() != CV_64F) {
    require(bytes_equal(actual, expected), "integer output differs: " +
                std::to_string(cv::norm(actual, expected, cv::NORM_INF)));
    return;
  }
  cv::Mat got64, expected64;
  actual.convertTo(got64, CV_64F);
  expected.convertTo(expected64, CV_64F);
  const double unit = expected.depth() == CV_32F ? 2e-5 : 4e-12;
  for (int y = 0; y < expected.rows; ++y) {
    for (int i = 0; i < expected.cols * expected.channels(); ++i) {
      const auto a = got64.ptr<double>(y)[i], b = expected64.ptr<double>(y)[i];
      if (a == b || (std::isnan(a) && std::isnan(b))) continue;
      require(std::isfinite(a) && std::isfinite(b) &&
                  std::abs(a-b) <= unit * std::max(1., std::abs(b)),
              "float output differs: " + std::to_string(a) + " vs " + std::to_string(b));
    }
  }
}

void initialize(Image& image, std::uint64_t seed) {
  cv::RNG rng(seed);
  double lo = -2, hi = 2;
  switch (image.host.depth()) {
    case CV_8U: lo = 0; hi = 256; break;
    case CV_16U: lo = 0; hi = 65536; break;
    case CV_16S: lo = -32768; hi = 32768; break;
    default: break;
  }
  rng.fill(image.host, cv::RNG::UNIFORM, lo, hi);
}

template<class Function>
void execute(Handle& handle, const Stream& upload, const Stream& use,
             Image& src, Image& dst, flagcv::HostTableData coefficients,
             const cv::Mat& expected, Function function) {
  Table table(coefficients, upload);
  src.upload(upload.value);
  dst.upload(upload.value);
  upload.wait();
  function(handle, src.view(), dst.view(), table.view(), use.value);
  cv::Mat got = dst.download(use);
  compare(got(dst.roi), expected);
  dst.pixels().copyTo(got(dst.roi));
  require(bytes_equal(got, dst.host), "destination guard bytes changed");
  require(bytes_equal(src.host, src.download(use)), "source bytes changed");
}

void gaussian_case(Handle& handle, const Stream& upload, const Stream& use, int depth,
                   int channels, Size image, const GaussianOptions& options, int border,
                   bool maximum = false, bool separable = false) {
  const int type = CV_MAKETYPE(depth, channels);
  Image src(image.height, image.width, type, 3, 2, 7);
  Image dst(image.height, image.width, type, 5, 1, 2);
  initialize(src, 413);
  initialize(dst, 91);
  if (maximum)
    src.pixels().setTo(cv::Scalar::all(depth == CV_8U ? 255 : 65535));
  const GaussianPlan plan(image, depth_for(depth), options);
  cv::Mat expected;
  cv::GaussianBlur(src.pixels(), expected, {options.kernel.width, options.kernel.height},
                   options.sigma_x, options.sigma_y, border | cv::BORDER_ISOLATED);
  std::unique_ptr<Image> scratch;
  if (separable) {
    const auto storage = plan.scratch_depth();
    const int temp_depth = storage == Depth::k32S ? CV_32S :
                           storage == Depth::k32F ? CV_32F : CV_64F;
    scratch = std::make_unique<Image>(image.height, image.width,
                CV_MAKETYPE(temp_depth, channels), 7, 1, 3);
    scratch->host.setTo(cv::Scalar::all(13));
    scratch->upload(upload.value);
  }
  execute(handle, upload, use, src, dst, plan.table(), expected,
          [&](auto& h, const auto& s, const auto& d, auto workspace, auto stream) {
            if (scratch)
              flagcv::GaussianBlur(h, s, d, plan, workspace, scratch->view(), stream);
            else
              flagcv::GaussianBlur(h, s, d, plan, workspace, stream);
          });
  if (scratch) {
    auto actual = scratch->download(use);
    scratch->pixels().copyTo(actual(scratch->roi));
    require(bytes_equal(actual, scratch->host), "GaussianBlur wrote outside its scratch ROI");
  }
}

void correlation_case(Handle& handle, const Stream& upload, const Stream& use,
                      Image& src, Image& dst, Size kernel, const double* weights,
                      Filter2DOptions options, int border) {
  const Filter2DPlan plan(kernel, weights, depth_for(src.host.depth()), options);
  const cv::Mat mask(kernel.height, kernel.width, CV_64F, const_cast<double*>(weights));
  cv::Mat expected;
  cv::filter2D(src.pixels(), expected, dst.host.depth(), mask,
               {options.anchor.x, options.anchor.y}, options.delta, border | cv::BORDER_ISOLATED);
  execute(handle, upload, use, src, dst, plan.table(), expected,
          [&](auto& h, const auto& s, const auto& d, auto workspace, auto stream) {
            flagcv::filter2D(h, s, d, plan, workspace, stream);
          });
}

template<class Pixel>
void half_weight_case(Handle& handle, const Stream& upload, const Stream& use,
                      int depth, int extent) {
  constexpr Size image{37, 19};
  const int center = extent / 2;
  std::vector<double> weights(extent * extent, 0.0);
  weights[center * extent + center] = 0.5;
  const Filter2DPlan plan({extent, extent}, weights.data(), depth_for(depth),
                          {{}, {center, center}, 0, Border::kReflect101});
  Image src(image.height, image.width, depth, 3, 1, 7);
  Image dst(image.height, image.width, depth, 4, 2, 5);
  src.host.setTo(cv::Scalar::all(31));
  for (const unsigned parity : {1u, 0u}) {
    dst.host.setTo(cv::Scalar::all(17));
    auto input = src.pixels();
    auto initial = dst.pixels();
    cv::Mat expected(image.height, image.width, depth);
    for (int y = 0; y < image.height; ++y) {
      for (int x = 0; x < image.width; ++x) {
        const unsigned p = static_cast<Pixel>(2 * (y * image.width + x) + parity);
        const unsigned q = p / 2;
        const auto rounded = static_cast<Pixel>(q + ((p & 1u) & (q & 1u)));
        input.ptr<Pixel>(y)[x] = static_cast<Pixel>(p);
        expected.ptr<Pixel>(y)[x] = rounded;
        initial.ptr<Pixel>(y)[x] = static_cast<Pixel>(~rounded);
      }
    }
    execute(handle, upload, use, src, dst, plan.table(), expected,
            [&](auto& h, const auto& s, const auto& d, auto workspace, auto stream) {
              flagcv::filter2D(h, s, d, plan, workspace, stream);
            });
  }
}

void filter_half_weight_reference(Handle& handle, const Stream& upload, const Stream& use) {
  // Only these four fixtures use the exact-half reference documented in numerical-tolerance.md.
  for (int extent : {13, 17})
    half_weight_case<std::uint8_t>(handle, upload, use, CV_8U, extent);
  for (int extent : {9, 11})
    half_weight_case<std::uint16_t>(handle, upload, use, CV_16U, extent);
  std::cout << "filter2D 8 exact half-weight fixtures passed\n";
}

void filter_three_tap_reference(Handle& handle, const Stream& upload, const Stream& use) {
  std::array<double, 81> weights{};
  weights[4 * 9 + 4] = 0.5;
  weights[4 * 9 + 5] = 0.25;
  weights[5 * 9 + 4] = 0.25;
  const Filter2DPlan plan({9, 9}, weights.data(), Depth::k16U,
                          {{}, {4, 4}, 0, Border::kReflect101});
  constexpr std::array<std::uint16_t, 13> extremes{
      0, 65535, 65534, 1, 2, 3, 65533, 65532, 4, 5, 6, 7, 65531};
  constexpr Size cases[] = {{37, 19}, {768, 512}, {9, 1}, {1, 9}, {1, 1}, {17, 13}};
  std::array<bool, 8> residues{};
  for (const Size size : cases) {
    Image src(size.height, size.width, CV_16UC1, 3, 1, 7);
    Image dst(size.height, size.width, CV_16UC1, 4, 2, 5);
    src.host.setTo(cv::Scalar::all(31));
    dst.host.setTo(cv::Scalar::all(17));
    auto input = src.pixels();
    auto initial = dst.pixels();
    if (size.width == 37 || size.width == 768) {
      cv::Mat packed(input.size(), input.type());
      cv::RNG rng(40738);
      rng.fill(packed, cv::RNG::UNIFORM, 0, 1024);
      packed.copyTo(input);
    } else if (size.width == 1 && size.height == 1) {
      input.ptr<std::uint16_t>(0)[0] = 65535;
    } else {
      for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
          input.ptr<std::uint16_t>(y)[x] = extremes[(y * size.width + x) % extremes.size()];
    }
    cv::Mat expected(input.size(), input.type());
    for (int y = 0; y < size.height; ++y) {
      const int below = y + 1 < size.height ? y + 1 : std::max(0, size.height - 2);
      for (int x = 0; x < size.width; ++x) {
        const int right = x + 1 < size.width ? x + 1 : std::max(0, size.width - 2);
        const auto* row = input.ptr<std::uint16_t>(y);
        const std::uint32_t total = 2u * row[x] + row[right] + input.ptr<std::uint16_t>(below)[x];
        residues[total % residues.size()] = true;
        const auto q = total / 4;
        const auto m = total % 4;
        const auto rounded = static_cast<std::uint16_t>(q + (m > 2 || (m == 2 && (q & 1u))));
        expected.ptr<std::uint16_t>(y)[x] = rounded;
        initial.ptr<std::uint16_t>(y)[x] = static_cast<std::uint16_t>(~rounded);
      }
    }
    execute(handle, upload, use, src, dst, plan.table(), expected,
            [&](auto& h, const auto& s, const auto& d, auto workspace, auto stream) {
              flagcv::filter2D(h, s, d, plan, workspace, stream);
            });
    std::cout << "filter2D exact three-tap " << size.height << 'x' << size.width << " passed\n";
  }
  require(std::all_of(residues.begin(), residues.end(), [](bool tested) { return tested; }),
          "three-tap fixtures must cover all remainders and both quotient parities");
}

bool gaussian_sigma_boundaries(Handle& handle, const Stream& upload, const Stream& use) {
  bool all_equal = true;
  for (const int depth : {CV_8U, CV_16U}) {
    const int bits = depth == CV_8U ? 8 : 16;
    const int units = 1 << bits;
    const double center = depth == CV_8U ? 0.39425309299377326 : 0.2771897452026664;
    for (const double sigma : {std::nextafter(center, 0.), center,
                                std::nextafter(center, std::numeric_limits<double>::infinity())}) {
      const GaussianOptions options{{3, 1}, sigma, 1, Border::kReplicate};
      const GaussianPlan plan({5, 1}, depth_for(depth), options);
      const auto* actual_table = static_cast<const double*>(plan.table().data);
      const auto axis = cv::getGaussianKernel(3, sigma, CV_64F);
      const double first = std::nearbyint(axis.at<double>(0) * units);
      const bool table_matches = actual_table[0] == first && actual_table[1] == units - 2*first &&
                                 actual_table[2] == first && actual_table[3] == units;
      Image src(1, 5, CV_MAKETYPE(depth, 1)), dst(1, 5, CV_MAKETYPE(depth, 1));
      src.host.setTo(0);
      src.host.col(2).setTo(units - 1);
      cv::Mat expected;
      cv::GaussianBlur(src.pixels(), expected, {3, 1}, sigma, 1, cv::BORDER_REPLICATE);
      const Table table(plan.table(), upload);
      src.upload(upload.value);
      upload.wait();
      flagcv::GaussianBlur(handle, src.view(), dst.view(), plan, table.view(), use.value);
      const cv::Mat got = dst.download(use);
      const bool gpu_matches = bytes_equal(got, expected);
      std::cout << "sigma-boundary " << bits << ' ' << std::setprecision(17) << sigma
                << " table_actual=" << actual_table[0] << ',' << actual_table[1] << ',' << actual_table[2]
                << " table_expected=" << first << ',' << units - 2*first << ',' << first
                << " gpu=" << got << " cpu=" << expected << '\n';
      all_equal &= table_matches && gpu_matches;
    }

    // Near-equal sigmas on equal axes reuse the horizontal table, even when
    // generating the vertical table independently crosses a rounding step.
    const double after = depth == CV_8U ? std::numeric_limits<double>::infinity() : 0.;
    const double neighbor = std::nextafter(center, after);
    const GaussianPlan plan({5, 5}, depth_for(depth), {{3, 3}, center, neighbor, Border::kReplicate});
    const auto* weights = static_cast<const double*>(plan.table().data);
    const bool shared = std::equal(weights, weights + 3, weights + 3);
    const Table table(plan.table(), upload);
    Image src(5, 5, CV_MAKETYPE(depth, 1)), dst(5, 5, CV_MAKETYPE(depth, 1));
    Image scratch(5, 5, CV_32SC1);
    src.host.setTo(0);
    src.host(cv::Rect(2, 2, 1, 1)).setTo(units - 1);
    src.upload(upload.value);
    upload.wait();
    cv::Mat expected;
    cv::GaussianBlur(src.pixels(), expected, {3, 3}, center, neighbor, cv::BORDER_REPLICATE);
    flagcv::GaussianBlur(handle, src.view(), dst.view(), plan, table.view(), use.value);
    const auto direct = dst.download(use);
    flagcv::GaussianBlur(handle, src.view(), dst.view(), plan, table.view(), scratch.view(), use.value);
    const auto separable = dst.download(use);
    auto center_value = [depth](const cv::Mat& values) -> int {
      return depth == CV_8U ? values.at<unsigned char>(2, 2) : values.at<unsigned short>(2, 2);
    };
    std::cout << "sigma-share " << bits << " x=" << weights[1] << " y=" << weights[4]
              << " direct=" << center_value(direct) << " two_pass=" << center_value(separable)
              << " cpu=" << center_value(expected) << '\n';
    all_equal &= shared && bytes_equal(direct, expected) && bytes_equal(separable, expected);
  }
  return all_equal;
}

template<class Callable>
double wall_us(Callable submit, const Stream& stream) {
  for (int i = 0; i < 8; ++i) submit();
  stream.wait();
  using Clock = std::chrono::steady_clock;
  const auto began = Clock::now();
  constexpr int calls = 16;
  for (int i = 0; i < calls; ++i) {
    submit();
    stream.wait();
  }
  return std::chrono::duration<double, std::micro>(Clock::now() - began).count() / calls;
}

void benchmark(Handle& handle, const Stream& stream) {
  struct Workload { const char* id; int h, w, kh, kw; bool gaussian; };
  constexpr Workload workloads[] = {
    {"gaussian_small3", 37, 65, 3, 3, true},
    {"gaussian_medium7", 512, 768, 7, 7, true},
    {"gaussian_large15", 720, 1280, 15, 15, true},
    {"gaussian_large35", 720, 1280, 3, 35, true},
    {"filter_small3", 37, 65, 3, 3, false},
    {"filter_medium5", 512, 768, 5, 5, false},
    {"filter_large9", 720, 1280, 9, 9, false}};
  for (const auto& task : workloads) {
    Image src(task.h, task.w, CV_32FC3), dst(task.h, task.w, CV_32FC3);
    initialize(src, 12951);
    src.upload(stream.value);
    cv::Mat expected;
    if (task.gaussian) {
      Image scratch(task.h, task.w, CV_32FC3);
      const GaussianPlan plan({task.w, task.h}, Depth::k32F, {{task.kw, task.kh}});
      const Table table(plan.table(), stream);
      cv::GaussianBlur(src.pixels(), expected, {task.kw, task.kh}, 0);
      auto fused = [&] { flagcv::GaussianBlur(handle, src.view(), dst.view(), plan, table.view(), stream.value); };
      auto separate = [&] {
        flagcv::GaussianBlur(handle, src.view(), dst.view(), plan, table.view(), scratch.view(), stream.value);
      };
      fused();
      compare(dst.download(stream), expected);
      separate();
      compare(dst.download(stream), expected);
      const auto bytes = std::int64_t{task.w} * task.h * 3 * sizeof(float);
      for (int block = 0; block < 7; ++block) {
        double direct = 0, separable = 0;
        if (block % 2) {
          separable = wall_us(separate, stream);
          direct = wall_us(fused, stream);
        } else {
          direct = wall_us(fused, stream);
          separable = wall_us(separate, stream);
        }
        std::cout << task.id << ',' << block << ',' << direct << ',' << separable << ',' << bytes << '\n';
      }
    } else {
      std::vector<double> weights(task.kh * task.kw);
      cv::Mat mask(task.kh, task.kw, CV_64F, weights.data());
      cv::RNG rng(525);
      rng.fill(mask, cv::RNG::UNIFORM, -.25, .25);
      const Filter2DPlan plan({task.kw, task.kh}, weights.data(), Depth::k32F,
                              {{}, {0, task.kh - 1}, .125, Border::kReflect101});
      const Table table(plan.table(), stream);
      cv::filter2D(src.pixels(), expected, -1, mask, {0,task.kh-1}, .125);
      auto submit = [&] { flagcv::filter2D(handle, src.view(), dst.view(), plan, table.view(), stream.value); };
      submit();
      compare(dst.download(stream), expected);
      for (int block = 0; block < 7; ++block)
        std::cout << task.id << ',' << block << ',' << wall_us(submit, stream) << ",,0\n";
    }
  }
}

void row_pair_cases(Handle& handle, const Stream& upload, const Stream& stream) {
  Image src(129, 257, CV_32FC3, 3, 2, 7), dst(129, 257, CV_32FC3, 4, 1, 9);
  initialize(src, 744);
  initialize(dst, 745);
  std::array<double, 24> weights{};
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = (i % 7 + 1) / 64.0 * (i % 2 == 0 ? 1 : -1);
  const Filter2DOptions options{{}, {5, 0}, -0.125, Border::kReflect101};
  correlation_case(handle, upload, stream, src, dst, {6, 4}, weights.data(),
                   options, cv::BORDER_REFLECT_101);
  // A coefficient below the normal F32 range keeps the zero-skipping direct kernel.
  weights[3] = std::numeric_limits<double>::denorm_min();
  correlation_case(handle, upload, stream, src, dst, {6, 4}, weights.data(),
                   options, cv::BORDER_REFLECT_101);
}

template<class Function>
void reject(Function function) {
  try { function(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid spatial filter request was accepted");
}

void rejection_cases(Handle& handle, const Stream& stream) {
  const Image src(9, 19, CV_8UC3), dst(9, 19, CV_8UC3), wrong(9, 19, CV_8UC1);
  const GaussianPlan gaussian({19, 9}, Depth::k8U, {{3, 3}});
  const Table table(gaussian.table(), stream);
  const auto bytes = table.view();
  const double one = 1;
  reject([&] { GaussianPlan({19, 9}, Depth::k8U, {{4, 3}}); });
  reject([&] { GaussianPlan({19, 9}, Depth::k8U, {{0, 3}, 1e100}); });
  reject([&] { GaussianPlan({19, 9}, Depth::k8U, {{3, 3}, 0, 0, Border::kWrap}); });
  reject([&] { Filter2DPlan({-1, 5}, &one, Depth::k8U); });
  reject([&] { Filter2DPlan({1, 1}, &one, Depth::k32F, {Depth::k16U}); });
  reject([&] { Filter2DPlan({1, 1}, &one, Depth::k8U, {{}, {1, 0}}); });
  double inf = std::numeric_limits<double>::infinity();
  reject([&] { Filter2DPlan({1, 1}, &inf, Depth::k8U); });
  reject([&] { flagcv::GaussianBlur(handle, src.view(), src.view(), gaussian, bytes); });
  reject([&] { flagcv::GaussianBlur(handle, src.view(), wrong.view(), gaussian, bytes); });
  reject([&] { flagcv::GaussianBlur(handle, src.view(), dst.view(), gaussian, {}); });
  const Image float_src(9, 19, CV_32FC3), float_dst(9, 19, CV_32FC3);
  const GaussianPlan floating({19, 9}, Depth::k32F, {{3, 3}});
  const Table float_table(floating.table(), stream);
  reject([&] {
    flagcv::GaussianBlur(handle, float_src.view(), float_dst.view(), floating,
                         float_table.view(), float_src.view(), stream.value);
  });
  reject([&] {
    flagcv::GaussianBlur(handle, float_src.view(), float_dst.view(), floating,
                         float_table.view(), float_dst.view(), stream.value);
  });
  reject([&] { flagcv::GaussianBlur(handle, src.view(), dst.view(), gaussian, {bytes.data, bytes.bytes-1}); });
  reject([&] { flagcv::GaussianBlur(handle, src.view(), dst.view(), gaussian, {dst.view().data(), bytes.bytes}); });
  reject([&] {
    flagcv::GaussianBlur(handle, src.view(), dst.view(), gaussian,
                         {static_cast<const char*>(bytes.data) + 1, bytes.bytes});
  });
}

}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef FLAGCV_TEST_HIP
    Handle handle("hip");
#else
    Handle handle;
#endif
    cv::setNumThreads(1);
    const Stream upload, default_stream(false), custom;
    if (argc == 2 && std::string_view(argv[1]) == "--gaussian-boundary")
      return gaussian_sigma_boundaries(handle, upload, custom) ? 0 : 1;
    if (argc == 2 && std::string_view(argv[1]) == "--filter2d-half-reference") {
      filter_half_weight_reference(handle, upload, custom);
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--filter2d-three-tap-reference") {
      filter_three_tap_reference(handle, upload, custom);
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
      std::cout << "case,block,direct_sync_us,separable_sync_us,scratch_bytes\n";
      benchmark(handle, custom);
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--row-pair") {
      row_pair_cases(handle, upload, custom);
      std::cout << "filter2D row-pair and rounded-zero fallback passed\n";
      return 0;
    }
    rejection_cases(handle, upload);

    int passed = 0;
    for (int depth : {CV_8U, CV_16U, CV_32F, CV_64F}) {
      for (const auto border : borders) {
        gaussian_case(handle, upload, custom, depth, 3, {31, 23}, {{3, 5}, 1.3, .7, border.flag}, border.cv);
        gaussian_case(handle, upload, default_stream, depth, 1, {7, 1}, {{5, 7}, 0, 0, border.flag}, border.cv);
        gaussian_case(handle, upload, custom, depth, 4, {39, 13}, {{0, 0}, 1.25, 0, border.flag}, border.cv);
        passed += 3;
      }
    }
    for (int depth : {CV_8U, CV_16U}) {
      gaussian_case(handle, upload, custom, depth, 2, {35, 19}, {{13, 11}, 2.13, 1.97}, cv::BORDER_REFLECT_101, true);
      ++passed;
    }
    gaussian_case(handle, upload, default_stream, CV_8U, 3, {31, 23},
                  {{-1, 3}, 1, 1}, cv::BORDER_REFLECT_101);
    gaussian_case(handle, upload, custom, CV_16U, 3, {37, 31},
                  {{35, 3}, 5.5, 1}, cv::BORDER_REFLECT_101, true);
    passed += 2;
    for (const int depth : {CV_8U, CV_16U, CV_32F, CV_64F}) {
      gaussian_case(handle, upload, custom, depth, 3, {37, 31},
                    {{7, 5}, 1.1, .65, Border::kConstant}, cv::BORDER_CONSTANT,
                    depth <= CV_16U, true);
      gaussian_case(handle, upload, custom, depth, 3, {37, 31},
                    {{35, 3}, 5.5, 1}, cv::BORDER_REFLECT_101, depth <= CV_16U, true);
      passed += 2;
    }
    require(gaussian_sigma_boundaries(handle, upload, custom), "GaussianBlur sigma boundary differs");
    passed += 8;
    std::cout << "GaussianBlur " << passed << " reference cases passed\n";

    constexpr double asymmetric[] = {0.5, -1.25, 0.0, 2.0, 0.25, -0.75};
    constexpr std::array<std::pair<int, int>, 13> pairs{{
      {CV_8U, CV_8U}, {CV_8U, CV_16U}, {CV_8U, CV_16S}, {CV_8U, CV_32F}, {CV_8U, CV_64F},
      {CV_16U, CV_16U}, {CV_16U, CV_32F}, {CV_16U, CV_64F},
      {CV_16S, CV_16S}, {CV_16S, CV_32F}, {CV_16S, CV_64F},
      {CV_32F, CV_32F}, {CV_64F, CV_64F}}};
    passed = 0;
    for (const auto& [sd, dd] : pairs) {
      for (const auto border : borders) {
        Image src(17, 33, CV_MAKETYPE(sd, 3), 3, 2, 5), dst(17, 33, CV_MAKETYPE(dd, 3), 4, 1, 7);
        initialize(src, 820);
        initialize(dst, 821);
        correlation_case(handle, upload, custom, src, dst, {3, 2}, asymmetric,
                         {depth_for(dd), {0, 1}, -0.5, border.flag}, border.cv);
        ++passed;
      }
    }
    Image u16(1, 2, CV_16UC1, 1, 1, 2), result(1, 2, CV_16UC1, 2, 1, 1);
    constexpr double overflow[] = {65536};
    u16.host.setTo(cv::Scalar::all(65535));
    correlation_case(handle, upload, custom, u16, result, {1, 1}, overflow, {{}, {}, 0, Border::kConstant}, 0);
    constexpr double contraction[] = {4097};
    u16.host.setTo(cv::Scalar::all(4097));
    correlation_case(handle, upload, custom, u16, result, {1, 1}, contraction,
                     {{}, {}, -16785408, Border::kConstant}, 0);
    constexpr double delta_first[] = {1, -16777216};
    u16.host.setTo(cv::Scalar::all(1));
    correlation_case(handle, upload, default_stream, u16, result, {2, 1}, delta_first,
                     {{}, {0, 0}, 16777216, Border::kConstant}, 0);
    constexpr double zero[] = {0};
    Image nonfinite(1, 2, CV_32FC1), float_result(1, 2, CV_32FC1);
    nonfinite.host.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    nonfinite.host.at<float>(0, 1) = 2;
    correlation_case(handle, upload, custom, nonfinite, float_result, {1, 1}, zero,
                     {{}, {}, 3, Border::kConstant}, 0);
    std::array<double, 81> larger{};
    larger[3] = 0.25;
    larger[32] = -0.125;
    larger[79] = 0.75;
    Image large_src(23, 37, CV_32FC3, 3, 1, 7), large_dst(23, 37, CV_32FC3, 4, 2, 5);
    initialize(large_src, 725);
    initialize(large_dst, 726);
    correlation_case(handle, upload, custom, large_src, large_dst, {9, 9}, larger.data(),
                     {{}, {1, 6}, .25, Border::kReflect101}, cv::BORDER_REFLECT_101);
    row_pair_cases(handle, upload, custom);
    std::cout << "filter2D " << passed + 7 << " reference cases passed\n";
    filter_half_weight_reference(handle, upload, custom);
    filter_three_tap_reference(handle, upload, custom);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
