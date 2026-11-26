// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "cuda_image.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <limits>
#include <iostream>
#include <fstream>
#include <future>
#include <sstream>
#include <memory>
#include <vector>
#include <thread>

namespace {
#ifdef FLAGCV_TEST_HIP
constexpr auto test_backend = FLAGCV_TEST_VENDOR;
#else
constexpr auto test_backend = "nvidia";
#endif

using namespace flagcv_test;

void fill_bytes(cv::Mat& image, cv::RNG& rng) {
  cv::Mat bytes(image.rows, static_cast<int>(image.cols * image.elemSize()), CV_8U,
                image.data, image.step);
  rng.fill(bytes, cv::RNG::UNIFORM, 0, 256);
}

void inputs(Op op, cv::Mat image, cv::RNG& rng) {
  if (arithmetic(op) && (image.depth() == CV_16F || image.depth() == CV_32F)) {
    cv::Mat values(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
    rng.fill(values, cv::RNG::UNIFORM, -13.0, 17.0);
    auto* first = values.ptr<float>();
    if (values.total() * values.channels() >= 4) {
      first[0] = -0.0f;
      first[1] = std::numeric_limits<float>::infinity();
      first[2] = std::numeric_limits<float>::quiet_NaN();
      first[3] = 1.0e-7f;
    }
    values.convertTo(image, image.type());
  } else {
    fill_bytes(image, rng);
  }
}

void check_guards(const Image& dst, cv::Mat received, const cv::Mat& mask) {
  for (int y = 0; y < received.rows; ++y) {
    for (int x = 0; x < received.cols; ++x) {
      const cv::Point pixel(x, y);
      if (dst.roi.contains(pixel) &&
          (mask.empty() || mask.ptr(y - dst.roi.y)[x - dst.roi.x] != 0))
        continue;
      const auto offset = x * received.elemSize();
      require(std::memcmp(received.ptr(y) + offset, dst.host.ptr(y) + offset,
                          received.elemSize()) == 0,
              "destination guard or unmasked pixel was changed");
    }
  }
}

enum class Mask { none, mixed, zero, full };
enum class Alias { none, lhs, rhs };

void check_case(flagcv::Handle& handle, Op op, int depth, int channels, int rows,
                int cols, Mask mask_pattern, Alias alias, bool custom, int code,
                double scale = 1.0, bool packed = false) {
  try {
    Stream stream(custom);
    const int type = CV_MAKETYPE(depth, channels);
    const int offset = packed ? 0 : 1;
    Image a(rows, cols, type, offset, offset, 13 * offset);
    Image b(rows, cols, type, 2 * offset, offset, 7 * offset);
    Image output(rows, cols, type, 3 * offset, 2 * offset, 29 * offset);
    Image mask(rows, cols, CV_8U, 1, 3, 6);
    Image& dst = alias == Alias::lhs ? a : alias == Alias::rhs ? b : output;
    const bool masked = mask_pattern != Mask::none;
    cv::RNG rng(0x317f);
    fill_bytes(a.host, rng);
    fill_bytes(b.host, rng);
    fill_bytes(mask.host, rng);
    inputs(op, a.pixels(), rng);
    inputs(op, b.pixels(), rng);
    fill_bytes(output.host, rng);
    auto mask_pixels = masked ? mask.pixels() : cv::Mat();
    if (masked) {
      if (mask_pattern == Mask::mixed) {
        rng.fill(mask_pixels, cv::RNG::UNIFORM, 0, 4);
        mask_pixels.ptr()[0] = 0;
        if (cols > 1) mask_pixels.ptr()[1] = 255;
      } else {
        mask_pixels.setTo(mask_pattern == Mask::full ? 255 : 0);
      }
      mask.upload(stream.value);
    }
    auto expected = dst.pixels().clone();
    reference(op, a.pixels(), b.pixels(), expected, code, mask_pixels, scale);
    // Poison each destination region outside exact aliasing and mask constraints.
    if (alias == Alias::none) {
      auto pixels = dst.pixels();
      const auto bytes = pixels.elemSize();
      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
          if (masked && mask_pixels.ptr(y)[x] == 0) continue;
          auto* address = pixels.ptr(y) + x * bytes;
          if (arithmetic(op) && (depth == CV_32F || depth == CV_16F)) {
            cv::Mat value(1, 1, type, address);
            value.setTo(cv::Scalar::all(53.0));
          } else {
            for (std::size_t i = 0; i < bytes; ++i)
              address[i] = static_cast<unsigned char>(~expected.ptr(y)[x * bytes + i]);
          }
        }
      }
    }
    a.upload(stream.value);
    if (!unary(op)) b.upload(stream.value);
    if (alias == Alias::none) dst.upload(stream.value);
    call(op, handle, a.view(), b.view(), dst.view(), stream.value, code,
         masked ? mask.view() : flagcv::GpuMat{}, scale);
    auto received = dst.download(stream);
    compare(op, received(dst.roi), expected);
    check_guards(dst, received, mask_pixels);
    if (alias != Alias::lhs) require(bytes_equal(a.host, a.download(stream)), "source A changed");
    if (!unary(op) && alias != Alias::rhs)
      require(bytes_equal(b.host, b.download(stream)), "source B changed");
    if (masked) require(bytes_equal(mask.host, mask.download(stream)), "mask changed");
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(name(op)) + " depth=" + std::to_string(depth) +
                             " channels=" + std::to_string(channels) +
                             " mask=" + std::to_string(static_cast<int>(mask_pattern)) +
                             " alias=" + std::to_string(static_cast<int>(alias)) + ": " + error.what());
  }
}

template <typename Function>
void rejected(Function function, const char* what) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(std::string("expected rejection for ") + what);
}

void validation(flagcv::Handle& handle) {
  Image a(7, 11, CV_8UC3, 0, 0, 3);
  Image b(7, 11, CV_8UC3);
  auto view = a.view();
  auto other = b.view();
  const flagcv::GpuMat shifted(static_cast<unsigned char*>(view.data()) + 1, 11,
                              7, view.depth(), 3, view.row_pitch());
  rejected([&] { flagcv::flip(handle, view, shifted, -1); }, "partly overlapping flip");
  rejected([&] { flagcv::add(handle, view, other, shifted); }, "partly overlapping add");
  rejected([&] { flagcv::add(handle, view, other, other, other); }, "mask channels");
  rejected([&] { flagcv::add(handle, view, other, {}); }, "empty output");
  rejected([&] { flagcv::multiply(handle, view, other, other, 0.5); }, "integer scale");
  const flagcv::GpuMat short_pitch(view.data(), 11, 7, view.depth(), 3, 32);
  rejected([&] { flagcv::bitwise_not(handle, short_pitch, other); }, "short pitch");
  const flagcv::GpuMat invalid_pitch(view.data(), 11, 7, flagcv::Depth::k32F, 3, 133);
  rejected([&] { flagcv::bitwise_not(handle, invalid_pitch, other); }, "odd 32F pitch");
  const flagcv::GpuMat overflow(view.data(), 11, 7, view.depth(), 3,
                                 std::numeric_limits<std::int64_t>::max());
  rejected([&] { flagcv::bitwise_not(handle, overflow, other); }, "overflowing view");
  const flagcv::GpuMat wrong_extent(view.data(), 10, 7, view.depth(), 3);
  rejected([&] { flagcv::subtract(handle, view, wrong_extent, other); }, "different size");
  const flagcv::GpuMat wrong_depth(view.data(), 11, 7, flagcv::Depth::k8S, 3);
  rejected([&] { flagcv::subtract(handle, view, wrong_depth, other); }, "different depth");
  const flagcv::GpuMat wrong_mask(view.data(), 10, 7, view.depth(), 1);
  rejected([&] { flagcv::bitwise_and(handle, view, other, other, wrong_mask); },
           "different mask size");
  const flagcv::GpuMat excessive_channels(view.data(), 1, 1, view.depth(), 5);
  rejected([&] { flagcv::flip(handle, excessive_channels, excessive_channels, 0); },
           "unsupported channel count");
}

void shared_storage(flagcv::Handle& handle) {
  Image storage(7, 18, CV_8UC3);
  storage.host.setTo(cv::Scalar(3, 19, 73));
  storage.host.colRange(6, 18).setTo(cv::Scalar::all(101));
  Image other(7, 6, CV_8UC3);
  other.host.setTo(cv::Scalar(1, 2, 3));
  auto* start = static_cast<unsigned char*>(storage.device.ptr);
  flagcv::GpuMat a(start, 6, 7, flagcv::Depth::k8U, 3, storage.host.step);
  flagcv::GpuMat dst(start + 18, 6, 7, flagcv::Depth::k8U, 3, storage.host.step);
  Stream stream;
  storage.upload(stream.value);
  other.upload(stream.value);
  flagcv::add(handle, a, other.view(), dst, stream.value);
  auto expected = storage.host.clone();
  auto expected_roi = expected.colRange(6, 12);
  cv::add(storage.host.colRange(0, 6), other.host, expected_roi);
  require(bytes_equal(storage.download(stream), expected), "disjoint sibling ROI");
}

void CUDART_CB hold_stream(void* context) {
  auto& release = *static_cast<std::atomic<bool>*>(context);
  while (!release.load()) std::this_thread::yield();
}

void stream_is_asynchronous(flagcv::Handle& handle) {
  Image a(4, 8, CV_8U), dst(4, 8, CV_8U);
  Stream stream;
  a.host.setTo(31);
  a.upload(stream.value);
  flagcv::bitwise_not(handle, a.view(), dst.view(), stream.value);
  stream.wait();
  std::atomic<bool> release{false};
  cuda_check(cudaLaunchHostFunc(stream.value, hold_stream, &release));
  const auto start = std::chrono::steady_clock::now();
  std::thread watchdog([&] {
    while (!release.load() &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    release.store(true);
  });
  try {
    flagcv::bitwise_not(handle, a.view(), dst.view(), stream.value);
    require(!release.load(), "cached call waited for caller's stream");
    require(cudaStreamQuery(stream.value) == cudaErrorNotReady,
            "custom stream should still be blocked");
  } catch (...) {
    release.store(true);
    watchdog.join();
    stream.wait();
    throw;
  }
  release.store(true);
  watchdog.join();
  const auto result = dst.download(stream);
  require(result.ptr()[0] == static_cast<unsigned char>(~31), "wrong stream result");
}

void multiply_u16_saturation(flagcv::Handle& handle) {
  const std::uint16_t left[] = {65535, 32768, 65535, 65535, 257, 256, 0, 42};
  const std::uint16_t right[] = {65535, 2, 1, 0, 255, 255, 65535, 100};
  const std::uint16_t expected[] = {65535, 65535, 65535, 0, 65535, 65280, 0, 4200};
  Stream stream;
  for (int width : {1, 8}) {
    Image a(1, width, CV_16UC1), b(1, width, CV_16UC1), dst(1, width, CV_16UC1);
    std::copy_n(left, width, a.host.ptr<std::uint16_t>());
    std::copy_n(right, width, b.host.ptr<std::uint16_t>());
    cv::Mat reference_output;
    reference(Op::multiply, a.host, b.host, reference_output);
    require(std::equal(expected, expected + width, reference_output.ptr<std::uint16_t>()),
            "U16C1 multiply reference must saturate for width " + std::to_string(width));

    a.upload(stream.value);
    b.upload(stream.value);
    flagcv::multiply(handle, a.view(), b.view(), dst.view(), stream.value);
    const auto actual = dst.download(stream);
    compare(Op::multiply, actual, reference_output);
  }
  std::cout << "U16C1 multiply saturation passed for 1x1 and 1x8\n";
}

constexpr std::array vector_ops{Op::add, Op::subtract, Op::multiply,
                               Op::bitwise_and, Op::bitwise_not, Op::flip};

void shared_semantic_vectors(flagcv::Handle& handle) {
  std::ifstream file(FLAGCV_TEST_DATA "/semantic_u8.csv");
  std::string row;
  require(static_cast<bool>(std::getline(file, row)), "missing shared semantics vectors");
  std::vector<std::array<int, 8>> vectors;
  while (std::getline(file, row)) {
    std::replace(row.begin(), row.end(), ',', ' ');
    std::istringstream fields(row);
    std::array<int, 8> values{};
    for (auto& value : values)
      require(static_cast<bool>(fields >> value), "invalid shared semantics vector");
    vectors.push_back(values);
  }
  const int cols = static_cast<int>(vectors.size());
  require(cols > 0, "empty shared semantics vectors");
  Stream stream;
  Image a(1, cols, CV_8U), b(1, cols, CV_8U), dst(1, cols, CV_8U);
  for (int col = 0; col < cols; ++col) {
    a.host.at<unsigned char>(0, col) = vectors[col][0];
    b.host.at<unsigned char>(0, col) = vectors[col][1];
  }
  a.upload(stream.value);
  b.upload(stream.value);
  for (std::size_t op = 0; op < vector_ops.size(); ++op) {
    call(vector_ops[op], handle, a.view(), b.view(), dst.view(), stream.value, 1);
    const auto actual = dst.download(stream);
    for (int col = 0; col < cols; ++col)
      require(actual.at<unsigned char>(0, col) == vectors[col][op + 2],
              std::string(name(vector_ops[op])) + " differs from shared integer vector");
  }
}

void cases(flagcv::Handle& handle) {
  multiply_u16_saturation(handle);
  shared_semantic_vectors(handle);
  // NVIDIA U8 BLOCK_X is 1024 for both pointwise and flip. Cross the tail
  // boundary in their packed and independently pitched paths.
  for (int width : {1023, 1024, 1025}) {
    for (bool packed : {false, true}) {
      check_case(handle, Op::add, CV_8U, 1, 3, width, Mask::none, Alias::none, true, 0,
                 1.0, packed);
      check_case(handle, Op::flip, CV_8U, 1, 3, width, Mask::none, Alias::none, false, 1,
                 1.0, packed);
    }
  }
  constexpr int all_depths[] = {CV_8U, CV_8S, CV_16U, CV_16S,
                                CV_32S, CV_16F, CV_32F, CV_64F};
  for (const auto op : operators) {
    for (int depth : all_depths) {
      if (arithmetic(op) && depth != CV_8U && depth != CV_16U &&
          depth != CV_16F && depth != CV_32F) continue;
      for (int channels : {1, 2, 3, 4}) {
        const auto mask = supports_mask(op) && channels >= 2 ? Mask::mixed : Mask::none;
        check_case(handle, op, depth, channels, 17, 67, mask, Alias::none,
                   channels != 1, channels - 2);
      }
    }
    check_case(handle, op, CV_8U, 3, 19, 173,
               supports_mask(op) ? Mask::mixed : Mask::none, Alias::lhs, true, -2);
    if (!unary(op))
      check_case(handle, op, CV_16U, 1, 5, 33,
                 supports_mask(op) ? Mask::mixed : Mask::none, Alias::rhs, false, 0);
    if (supports_mask(op)) {
      check_case(handle, op, CV_8U, 1, 3, 33, Mask::zero, Alias::none, true, 0);
      check_case(handle, op, CV_8U, 3, 3, 33, Mask::full, Alias::lhs, false, 0);
    }
  }
  for (int code : {-7, 0, 9}) {
    check_case(handle, Op::flip, CV_8U, 3, 1, 19, Mask::none, Alias::lhs, true, code);
    check_case(handle, Op::flip, CV_8U, 3, 19, 1, Mask::none, Alias::none, false, code);
  }
  check_case(handle, Op::flip, CV_64F, 4, 1, 1, Mask::none, Alias::lhs, false, -2);
  check_case(handle, Op::multiply, CV_32F, 1, 19, 67, Mask::none, Alias::none, true, -0.25);
  check_case(handle, Op::multiply, CV_32F, 1, 17, 67, Mask::none, Alias::none, true, 0.75);
  // The NVIDIA launch folds rows into grid Z when CUDA's Y limit is reached.
  check_case(handle, Op::flip, CV_8U, 3, 65539, 1, Mask::none, Alias::none, true, -1);
  validation(handle);
  shared_storage(handle);
  stream_is_asynchronous(handle);
}

void platform_failure() {
  cuda_check(cudaSetDevice(0));
  flagcv::Handle handle(test_backend);
  Stream stream;
  Image src(4, 8, CV_8U), dst(4, 8, CV_8U);
  src.host.setTo(13);
  dst.host.setTo(27);
  src.upload(stream.value);
  dst.upload(stream.value);
  bool failed = false;
  try {
    flagcv::add(handle, src.view(), src.view(), dst.view(), stream.value);
  } catch (const flagcv::Error& error) {
    failed = std::string(error.what()).find("flagcv_missing_platform_entry") != std::string::npos;
  }
  require(failed, "registered platform kernel failure must surface to the caller");
  require(bytes_equal(dst.download(stream), dst.host), "failed dispatch wrote output");
  flagcv::bitwise_not(handle, src.view(), dst.view(), stream.value);
  require(dst.download(stream).ptr()[0] == static_cast<unsigned char>(~13),
          "unregistered operator failed to select the common implementation");
}

void independent_handles(int device) {
  auto exercise = [device](flagcv::Handle& handle, unsigned char pixel) {
    cuda_check(cudaSetDevice(device));
    Stream stream;
    Image src(5, 32, CV_8U), dst(5, 32, CV_8U);
    src.host.setTo(pixel);
    src.upload(stream.value);
    flagcv::bitwise_not(handle, src.view(), dst.view(), stream.value);
    const auto output = dst.download(stream);
    require(cv::countNonZero(output != static_cast<unsigned char>(~pixel)) == 0,
            "concurrent handle returned the wrong output");
  };
  flagcv::Handle survivor(test_backend, device);
  auto first = std::async(std::launch::async, [&] {
    flagcv::Handle transient(test_backend, device);
    exercise(transient, 13);
  });
  auto second = std::async(std::launch::async, [&] { exercise(survivor, 42); });
  first.get();
  second.get();
  exercise(survivor, 77);
}

void across_devices(int device_count) {
  std::vector<std::unique_ptr<flagcv::Handle>> handles;
  for (int device = 0; device < device_count; ++device)
    handles.push_back(std::make_unique<flagcv::Handle>(test_backend, device));
  for (int device = 0; device < device_count; ++device) {
    cuda_check(cudaSetDevice(device));
    auto& handle = *handles[device];
    Image src(4, 8, CV_8U), dst(4, 8, CV_8U);
    Stream stream;
    src.host.setTo(19 + device);
    src.upload(stream.value);
    const int other = (device + 1) % device_count;
    cuda_check(cudaSetDevice(other));
    flagcv::bitwise_not(handle, src.view(), dst.view(), stream.value);
    int current = -1;
    cuda_check(cudaGetDevice(&current));
    require(current == other, "dispatch changed the caller's current device");
    cuda_check(cudaSetDevice(device));
    require(dst.download(stream).ptr()[0] == static_cast<unsigned char>(~(19 + device)),
            "handle used the wrong device");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    configure_cpu_reference();
    if (argc == 2 && std::string(argv[1]) == "--platform-failure") {
      platform_failure();
      return 0;
    }
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count));
    const int requested = argc == 2 ? std::stoi(argv[1]) : 1;
    require(requested >= 1 && requested <= count, "requested GPUs unavailable");
    for (int device = 0; device < requested; ++device) {
      cuda_check(cudaSetDevice(device));
      flagcv::Handle handle(test_backend, device);
      cases(handle);
      independent_handles(device);
      std::cout << test_backend << " device " << device << " passed public API matrix\n";
    }
    across_devices(requested);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
