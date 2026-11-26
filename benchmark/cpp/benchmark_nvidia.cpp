// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <vector>

#include "tests/cpp/cuda_image.hpp"

namespace {
using namespace flagcv_test;

class Event {
 public:
  Event() { cuda_check(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  void record(cudaStream_t stream) { cuda_check(cudaEventRecord(event_, stream)); }
  void wait() const { cuda_check(cudaEventSynchronize(event_)); }
  double elapsed(const Event& end) const {
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, event_, end.event_));
    return ms;
  }

 private:
  cudaEvent_t event_;
};

struct Timing {
  double stream_us, submit_us;
};

struct Case {
  int height, width, depth, channels, code;
  bool roi = false;
};

class OpenCvOperation {
 public:
  OpenCvOperation(Op op, const Image& a, const Image& b, const Image& dst,
                  cudaStream_t stream, int flip_code)
      : op_(op),
        a_(header(a)), b_(header(b)), dst_(header(dst)),
        stream_(cv::cuda::StreamAccessor::wrapStream(stream)), code_(flip_code) {}

  void run() {
    switch (op_) {
      case Op::add: cv::cuda::add(a_, b_, dst_, cv::noArray(), -1, stream_); break;
      case Op::subtract: cv::cuda::subtract(a_, b_, dst_, cv::noArray(), -1, stream_); break;
      case Op::multiply: cv::cuda::multiply(a_, b_, dst_, 1.0, -1, stream_); break;
      case Op::bitwise_and: cv::cuda::bitwise_and(a_, b_, dst_, cv::noArray(), stream_); break;
      case Op::bitwise_not: cv::cuda::bitwise_not(a_, dst_, cv::noArray(), stream_); break;
      case Op::flip: cv::cuda::flip(a_, dst_, code_, stream_); break;
    }
  }

  unsigned char* output() const { return dst_.data; }

 private:
  static cv::cuda::GpuMat header(const Image& image) {
    const auto view = image.view();
    return {view.height(), view.width(), image.host.type(), view.data(),
            static_cast<std::size_t>(view.row_pitch())};
  }

  Op op_;
  cv::cuda::GpuMat a_, b_, dst_;
  cv::cuda::Stream stream_;
  int code_;
};

template <typename Function>
Timing measure(Function function, cudaStream_t stream, int repetitions) {
  using Clock = std::chrono::steady_clock;
  Event start, end;
  start.record(stream);
  const auto host_start = Clock::now();
  for (int i = 0; i < repetitions; ++i) function();
  const auto host_end = Clock::now();
  end.record(stream);
  end.wait();
  return {1000 * start.elapsed(end) / repetitions,
          std::chrono::duration<double, std::micro>(host_end - host_start).count() /
              repetitions};
}

// OpenCV perf uses cv::getTickCount. Drain only the tested stream inside
// the measured interval so the public async call's result is ready at the end.
template <typename Function>
double synchronized_us(Function function, const Stream& stream, int repetitions) {
  const auto start = cv::getTickCount();
  for (int i = 0; i < repetitions; ++i) function();
  stream.wait();
  return (cv::getTickCount() - start) * 1e6 / cv::getTickFrequency() / repetitions;
}

std::string sample_field(const std::vector<double>& samples) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) out << ';';
    out << samples[i];
  }
  return out.str();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void fill(Op op, Image& image, cv::RNG& rng) {
  auto values = image.pixels();
  rng.fill(values, cv::RNG::UNIFORM, image.host.depth() == CV_32F ? -13 : 0,
           image.host.depth() == CV_32F ? 19 :
           image.host.depth() == CV_16U ? (op == Op::multiply ? 32768 : 65536) : 256);
  if (op == Op::flip && image.host.depth() == CV_32F) values.ptr<float>()[0] = -0.f;
}

void verify_guards(const Image& image, cv::Mat result) {
  const auto row_bytes = image.roi.width * result.elemSize();
  auto target = result(image.roi);
  const auto original = image.pixels();
  for (int row = 0; row < target.rows; ++row)
    std::memcpy(target.ptr(row), original.ptr(row), row_bytes);
  require(bytes_equal(image.host, result), "wrote beyond output view");
}

void benchmark(Op op, const Case& test, flagcv::Handle& handle, const Stream& stream,
               std::ostream& out, int samples) {
  const int type = CV_MAKETYPE(test.depth, test.channels);
  const int x = test.roi ? 1 : 0, y = test.roi ? 1 : 0;
  Image a(test.height, test.width, type, x, y, test.roi ? 23 : 0);
  Image b(test.height, test.width, type, x * 2, y, test.roi ? 13 : 0);
  Image dst(test.height, test.width, type, x * 3, y, test.roi ? 31 : 0);
  Image baseline(test.height, test.width, type, x * 3, y, test.roi ? 31 : 0);
  cv::RNG rng(1664525);
  fill(op, a, rng);
  if (!unary(op)) fill(op, b, rng);
  cv::Mat expected;
  reference(op, a.pixels(), b.pixels(), expected, test.code);
  dst.host.setTo(cv::Scalar::all(37));
  baseline.host.setTo(cv::Scalar::all(37));
  if (test.depth == CV_32F) {
    auto target = dst.pixels();
    target.setTo(cv::Scalar::all(911.0));
  } else {
    auto target = dst.pixels();
    cv::bitwise_not(expected, target);
  }
  dst.host.copyTo(baseline.host);
  a.upload(stream.value);
  if (!unary(op)) b.upload(stream.value);
  dst.upload(stream.value);
  baseline.upload(stream.value);
  const auto av = a.view(), bv = b.view(), dv = dst.view();
  const auto candidate = [&] { call(op, handle, av, bv, dv, stream.value, test.code); };

  OpenCvOperation cv_op(op, a, b, baseline, stream.value, test.code);
  const auto native = [&] { cv_op.run(); };
  candidate();
  native();
  auto result = dst.download(stream);
  auto cv_result = baseline.download(stream);
  require(cv_op.output() == baseline.view().data(), "OpenCV replaced preallocated dst");
  try {
    compare(op, result(dst.roi), expected);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("FlagCV: ") + error.what());
  }
  try {
    compare(op, cv_result(baseline.roi), expected);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("OpenCV CUDA: ") + error.what());
  }
  verify_guards(dst, result);
  verify_guards(baseline, cv_result);
  require(bytes_equal(a.download(stream), a.host), "input A changed");
  if (!unary(op)) require(bytes_equal(b.download(stream), b.host), "input B changed");

  for (int i = 0; i < 20; ++i) {
    candidate();
    native();
  }
  stream.wait();
  const int iterations = std::clamp(16000000 / (test.height * test.width), 20, 200);
  std::vector<double> cand_stream, cv_stream, cand_host, cv_host;
  std::vector<double> cand_single, cv_single, cand_batch, cv_batch;
  for (int sample = 0; sample < samples; ++sample) {
    Timing cand_time{}, cv_time{};
    if (sample % 2 == 0) {
      cand_time = measure(candidate, stream.value, iterations);
      cv_time = measure(native, stream.value, iterations);
      cand_single.push_back(synchronized_us(candidate, stream, 1));
      cv_single.push_back(synchronized_us(native, stream, 1));
      cand_batch.push_back(synchronized_us(candidate, stream, iterations));
      cv_batch.push_back(synchronized_us(native, stream, iterations));
    } else {
      cv_time = measure(native, stream.value, iterations);
      cand_time = measure(candidate, stream.value, iterations);
      cv_single.push_back(synchronized_us(native, stream, 1));
      cand_single.push_back(synchronized_us(candidate, stream, 1));
      cv_batch.push_back(synchronized_us(native, stream, iterations));
      cand_batch.push_back(synchronized_us(candidate, stream, iterations));
    }
    cand_stream.push_back(cand_time.stream_us);
    cv_stream.push_back(cv_time.stream_us);
    cand_host.push_back(cand_time.submit_us);
    cv_host.push_back(cv_time.submit_us);
  }
  const double cand = median(cand_stream), base = median(cv_stream);
  const char* depth = test.depth == CV_32F ? "32F" : test.depth == CV_16U ? "16U" : "8U";
  out << name(op) << ',' << test.height << ',' << test.width << ',' << depth
      << ',' << test.channels << ',' << (test.roi ? "roi" : "packed")
      << ',' << test.code << ',' << samples << ',' << iterations << ','
      << cand << ',' << base << ',' << base / cand << ','
      << median(cand_host) << ',' << median(cv_host) << ",true,"
      << (op == Op::multiply && test.depth == CV_16U ? "u16_product_in_i32" : "default")
      << ',' << median(cand_single) << ',' << median(cv_single)
      << ',' << median(cand_batch) << ',' << median(cv_batch)
      << ',' << sample_field(cand_single) << ',' << sample_field(cv_single)
      << ',' << sample_field(cand_batch) << ',' << sample_field(cv_batch) << '\n';
  out.flush();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: flagcv_benchmark_nvidia DEVICE OUTPUT.csv [ODD_SAMPLE_COUNT]\n";
    return 2;
  }
  try {
    configure_cpu_reference();
    const int device = std::stoi(argv[1]);
    const int samples = argc == 4 ? std::stoi(argv[3]) : 7;
    require(samples >= 3 && samples % 2 != 0, "expected an odd sample count >= 3");
    cuda_check(cudaSetDevice(device));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));
    char pci_bus[32];
    cuda_check(cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), device));
    int driver = 0, runtime = 0;
    cuda_check(cudaDriverGetVersion(&driver));
    cuda_check(cudaRuntimeGetVersion(&runtime));
    std::ofstream out(argv[2]);
    require(static_cast<bool>(out), "cannot open benchmark output");
    out << "# device=" << properties.name << ",ordinal=" << device
        << ",pci_bus=" << pci_bus << ",driver_api=" << driver
        << ",cuda_runtime=" << runtime << ",opencv=" << CV_VERSION
        << ",protocol=flagcv-cpp-opencv-ticks-stream-wait/2" << '\n'
        << "op,height,width,depth,channels,layout,flip_code,samples,iterations,"
           "flagcv_stream_us,opencv_stream_us,speedup,flagcv_submit_us,opencv_submit_us,"
           "validated_against_opencv_cpu,input_domain,"
           "flagcv_host_single_us,opencv_host_single_us,flagcv_host_batch_us,opencv_host_batch_us,"
           "flagcv_host_single_samples_us,opencv_host_single_samples_us,"
           "flagcv_host_batch_samples_us,opencv_host_batch_samples_us\n"
        << std::fixed << std::setprecision(5);
    flagcv::Handle handle("nvidia", device);
    Stream stream;
    for (auto op : operators) {
      for (auto shape : {cv::Size(64, 64), cv::Size(67, 61), cv::Size(640, 480),
                         cv::Size(1920, 1080), cv::Size(3840, 2160)}) {
        for (int depth : {CV_8U, CV_16U, CV_32F}) {
          if (depth != CV_8U && !arithmetic(op) && op != Op::flip) continue;
          for (int code : (op == Op::flip ? std::vector{-1, 0, 1}
                                           : std::vector{0})) {
            benchmark(op, {shape.height, shape.width, depth, 1, code},
                      handle, stream, out, samples);
          }
        }
      }
      for (int channels : {3, 4}) {
        benchmark(op, {513, 1021, CV_8U, channels, op == Op::flip ? -1 : 0, true},
                  handle, stream, out, samples);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
