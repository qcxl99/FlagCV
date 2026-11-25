// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "cuda_image.hpp"

#include <atomic>
#include <chrono>
#include <limits>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>

namespace {
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

void check_case(flagcv::Handle& handle, Op op, int depth, int channels, int rows,
                int cols, bool masked, bool in_place, bool custom, int code,
                double scale = 1.0) {
  try {
    Stream stream(custom);
    const int type = CV_MAKETYPE(depth, channels);
    Image a(rows, cols, type, 1, 1, 13);
    Image b(rows, cols, type, 2, 1, 7);
    Image output(rows, cols, type, 3, 2, 29);
    Image mask(rows, cols, CV_8U, 1, 3, 6);
    Image& dst = in_place ? a : output;
    cv::RNG rng(0x317f);
    fill_bytes(a.host, rng);
    fill_bytes(b.host, rng);
    fill_bytes(mask.host, rng);
    inputs(op, a.pixels(), rng);
    inputs(op, b.pixels(), rng);
    fill_bytes(output.host, rng);
    auto mask_pixels = masked ? mask.pixels() : cv::Mat();
    if (masked) {
      rng.fill(mask_pixels, cv::RNG::UNIFORM, 0, 4);
      mask_pixels.ptr()[0] = 0;
      if (cols > 1) mask_pixels.ptr()[1] = 255;
      mask.upload(stream.value);
    }
    auto expected = dst.pixels().clone();
    reference(op, a.pixels(), b.pixels(), expected, code, mask_pixels, scale);
    // Poison each destination region outside exact aliasing and mask constraints.
    if (!in_place) {
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
    if (!in_place) dst.upload(stream.value);
    call(op, handle, a.view(), b.view(), dst.view(), stream.value, code,
         masked ? mask.view() : flagcv::GpuMat{}, scale);
    auto received = dst.download(stream);
    compare(op, received(dst.roi), expected);
    check_guards(dst, received, mask_pixels);
    if (!in_place) require(bytes_equal(a.host, a.download(stream)), "source A changed");
    if (!unary(op)) require(bytes_equal(b.host, b.download(stream)), "source B changed");
    if (masked) require(bytes_equal(mask.host, mask.download(stream)), "mask changed");
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(name(op)) + " depth=" + std::to_string(depth) +
                             " channels=" + std::to_string(channels) +
                             " mask=" + std::to_string(masked) +
                             " alias=" + std::to_string(in_place) + ": " + error.what());
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

void cases(flagcv::Handle& handle) {
  constexpr int all_depths[] = {CV_8U, CV_8S, CV_16U, CV_16S,
                                CV_32S, CV_16F, CV_32F, CV_64F};
  for (const auto op : operators) {
    for (int depth : all_depths) {
      if (arithmetic(op) && depth != CV_8U && depth != CV_16U &&
          depth != CV_16F && depth != CV_32F) continue;
      for (int channels : {1, 2, 3, 4}) {
        const bool mask = supports_mask(op) && channels >= 2;
        check_case(handle, op, depth, channels, 17, 67, mask, false,
                   channels != 1, channels - 2);
      }
    }
    check_case(handle, op, CV_8U, 3, 19, 173, supports_mask(op), true, true, -2);
  }
  for (int code : {-7, 0, 9}) {
    check_case(handle, Op::flip, CV_8U, 3, 1, 19, false, true, true, code);
    check_case(handle, Op::flip, CV_8U, 3, 19, 1, false, false, false, code);
  }
  check_case(handle, Op::flip, CV_64F, 4, 1, 1, false, true, false, -2);
  check_case(handle, Op::multiply, CV_32F, 1, 19, 67, false, false, true, -0.25);
  check_case(handle, Op::multiply, CV_32F, 1, 17, 67, false, false, true, 0.75);
  // The NVIDIA launch folds rows into grid Z when CUDA's Y limit is reached.
  check_case(handle, Op::flip, CV_8U, 3, 65539, 1, false, false, true, -1);
  validation(handle);
  shared_storage(handle);
  stream_is_asynchronous(handle);
}

void platform_failure() {
  cuda_check(cudaSetDevice(0));
  flagcv::Handle handle;
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
    failed = std::string(error.what()).find("flagcv_missing_nvidia_entry") != std::string::npos;
  }
  require(failed, "registered platform kernel failure must surface to the caller");
  require(bytes_equal(dst.download(stream), dst.host), "failed dispatch wrote output");
  flagcv::bitwise_not(handle, src.view(), dst.view(), stream.value);
  require(dst.download(stream).ptr()[0] == static_cast<unsigned char>(~13),
          "unregistered operator failed to select the common implementation");
}

void across_devices(int device_count) {
  std::vector<std::unique_ptr<flagcv::Handle>> handles;
  for (int device = 0; device < device_count; ++device)
    handles.push_back(std::make_unique<flagcv::Handle>("nvidia", device));
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
    require(requested >= 1 && requested <= count, "requested NVIDIA GPUs unavailable");
    for (int device = 0; device < requested; ++device) {
      cuda_check(cudaSetDevice(device));
      flagcv::Handle handle("nvidia", device);
      cases(handle);
      std::cout << "NVIDIA device " << device << " passed public API matrix\n";
    }
    across_devices(requested);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
