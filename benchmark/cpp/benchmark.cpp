// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifdef FLAGCV_NATIVE_OPENCV_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "tests/cpp/cuda_image.hpp"

namespace {
using namespace flagcv_test;

class Event {
 public:
  Event() { cuda_check(cudaEventCreate(&event_)); }
  ~Event() { (void)cudaEventDestroy(event_); }
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

struct Case {
  Op op;
  int height, width, depth, channels, code;
  bool roi;
  const char* input_domain;
  const char* id;
};

#include "core_cases.hpp"

#ifdef FLAGCV_NATIVE_OPENCV_CUDA
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

#endif

struct Samples {
  std::vector<double> device, api, submit, batch;
};

template <typename Run>
void measure(Run run, const Stream& stream, Event& start, Event& end,
             Samples& samples, bool diagnostics) {
  using Clock = std::chrono::steady_clock;
  using Microseconds = std::chrono::duration<double, std::micro>;
  stream.wait();
  start.record(stream.value);
  run();
  end.record(stream.value);
  end.wait();
  samples.device.push_back(1000 * start.elapsed(end));

  stream.wait();
  const auto api_start = Clock::now();
  run();
  const auto submit_end = diagnostics ? Clock::now() : api_start;
  stream.wait();
  samples.api.push_back(Microseconds(Clock::now() - api_start).count());
  if (diagnostics) {
    samples.submit.push_back(Microseconds(submit_end - api_start).count());
    constexpr int iterations = 60;
    const auto batch_start = Clock::now();
    for (int i = 0; i < iterations; ++i) run();
    stream.wait();
    samples.batch.push_back(Microseconds(Clock::now() - batch_start).count() / iterations);
  }
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

void record(std::ostream& out, const Case& test, const char* api,
            const Samples& samples, const Samples& native, const char* native_status) {
  const char* depth = test.depth == CV_32F ? "32F" : test.depth == CV_16U ? "16U" : "8U";
  out << test.id << ',' << name(test.op) << ',' << test.height << ',' << test.width
      << ',' << depth << ',' << test.channels << ',' << (test.roi ? "roi" : "packed")
      << ',' << test.code << ',' << test.input_domain << ',' << api << ",true," << native_status;
  for (const auto* series : {&samples.device, &samples.api, &samples.submit, &samples.batch,
                             &native.device, &native.api}) {
    out << ',';
    if (!series->empty()) out << median(*series);
    out << ',' << sample_field(*series);
  }
  out << '\n';
  out.flush();
}

#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
bool compare_convenience(const Case& test) {
  return test.height == 64 && test.width == 64 && test.depth == CV_8U
      && (test.op == Op::add || test.op == Op::bitwise_not
          || (test.op == Op::flip && test.code == 1));
}
#endif

void benchmark(const Case& test, flagcv::Handle& handle, const Stream& stream,
               std::ostream& out, int count, bool diagnostics) {
  const auto op = test.op;
  const int type = CV_MAKETYPE(test.depth, test.channels);
  const int xy = test.roi ? 1 : 0;
  Image a(test.height, test.width, type, xy, xy, 23 * xy);
  Image b(test.height, test.width, type, 2 * xy, xy, 13 * xy);
  Image dst(test.height, test.width, type, 3 * xy, xy, 31 * xy);
  cv::RNG rng(1664525);
  fill(op, a, rng);
  if (!unary(op)) fill(op, b, rng);
  cv::Mat expected;
  reference(op, a.pixels(), b.pixels(), expected, test.code);
  dst.host.setTo(cv::Scalar::all(37));
  auto target = dst.pixels();
  if (test.depth == CV_32F) target.setTo(cv::Scalar::all(911.0));
  else cv::bitwise_not(expected, target);

  a.upload(stream.value);
  if (!unary(op)) b.upload(stream.value);
  dst.upload(stream.value);
  const auto av = a.view(), bv = b.view(), dv = dst.view();
  const auto candidate = [&] { call(op, handle, av, bv, dv, stream.value, test.code); };
  const auto verify = [&](const Image& image) {
    const auto result = image.download(stream);
    compare(op, result(image.roi), expected);
    verify_guards(image, result);
  };
  candidate();
  verify(dst);
  require(bytes_equal(a.download(stream), a.host), "input A changed");
  if (!unary(op)) require(bytes_equal(b.download(stream), b.host), "input B changed");

#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
  const bool use_convenience = compare_convenience(test);
  const auto convenience = [&] {
    switch (op) {
      case Op::add: flagcv::add(av, bv, dv, stream.value); break;
      case Op::bitwise_not: flagcv::bitwise_not(av, dv, stream.value); break;
      case Op::flip: flagcv::flip(av, dv, test.code, stream.value); break;
      default: throw std::logic_error("no convenience comparison configured");
    }
  };
  if (use_convenience) {
    dst.upload(stream.value);
    convenience();
    verify(dst);
  }
#endif

  const char* native_status = "unavailable";
#ifdef FLAGCV_NATIVE_OPENCV_CUDA
  Image baseline(test.height, test.width, type, 3 * xy, xy, 31 * xy);
  dst.host.copyTo(baseline.host);
  std::optional<OpenCvOperation> native_op;
  try {
    baseline.upload(stream.value);
    native_op.emplace(op, a, b, baseline, stream.value, test.code);
    native_op->run();
    verify(baseline);
    require(native_op->output() == baseline.view().data(), "OpenCV replaced preallocated dst");
    for (int i = 0; i < 20; ++i) native_op->run();
    stream.wait();
    native_status = "passed";
  } catch (const std::exception& error) {
    stream.wait();
    native_status = "failed";
    native_op.reset();
    std::cerr << test.id << " OpenCV CUDA: " << error.what() << '\n';
  }
#endif
  for (int i = 0; i < 20; ++i) {
    candidate();
#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
    if (use_convenience) convenience();
#endif
  }
  stream.wait();

  Samples explicit_samples, convenience_samples, native_samples;
  Event start, end;
  const auto time_secondary = [&] {
#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
    if (use_convenience) measure(convenience, stream, start, end, convenience_samples, diagnostics);
#endif
#ifdef FLAGCV_NATIVE_OPENCV_CUDA
    if (native_op) measure([&] { native_op->run(); }, stream, start, end, native_samples, false);
#endif
  };
  for (int sample = 0; sample < count; ++sample) {
    if (sample % 2 != 0) time_secondary();
    measure(candidate, stream, start, end, explicit_samples, diagnostics);
    if (sample % 2 == 0) time_secondary();
  }
  record(out, test, "explicit", explicit_samples, native_samples, native_status);
#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
  if (use_convenience)
    record(out, test, "setDevice", convenience_samples, native_samples, native_status);
#endif
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: flagcv_benchmark_" FLAGCV_BENCHMARK_VENDOR
                 " DEVICE OUTPUT.csv [ODD_SAMPLE_COUNT] [--diagnostics]\n";
    return 2;
  }
  try {
    configure_cpu_reference();
    const int device = std::stoi(argv[1]);
    const int samples = argc >= 4 ? std::stoi(argv[3]) : 21;
    require(samples >= 3 && samples % 2 != 0, "expected an odd sample count >= 3");
    const bool diagnostics = argc == 5;
    if (diagnostics) require(std::string_view(argv[4]) == "--diagnostics", "unknown argument");
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
    out << "# device=" << properties.name << ",backend=" FLAGCV_BENCHMARK_VENDOR
        << ",ordinal=" << device << ",pci_bus=" << pci_bus << ",driver_api=" << driver
        << ",runtime_api=" << runtime << ",opencv=" << CV_VERSION
        << ",protocol=flagcv-cpp-single-stream-event-and-api/1,samples=" << samples << '\n'
        << "case_id,op,height,width,depth,channels,layout,flip_code,input_domain,api,"
           "validated_against_opencv_cpu,native_status,device_us,device_samples_us,api_us,"
           "api_samples_us,submit_us,submit_samples_us,batch_us,batch_samples_us,"
           "native_device_us,native_device_samples_us,native_api_us,native_api_samples_us\n"
        << std::fixed << std::setprecision(5);
    flagcv::Handle handle(FLAGCV_BENCHMARK_VENDOR, device);
#ifndef FLAGCV_BENCHMARK_EXPLICIT_ONLY
    flagcv::setDevice(FLAGCV_BENCHMARK_VENDOR, device);
#endif
    Stream stream;
    for (const auto& test : core_cases) {
      try {
        benchmark(test, handle, stream, out, samples, diagnostics);
      } catch (const std::exception& error) {
        throw std::runtime_error(std::string(test.id) + ": " + error.what());
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
