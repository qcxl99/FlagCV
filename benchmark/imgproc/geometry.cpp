#define main small_fixture_main
#include "test_projective.cpp"
#undef main
#include "live_device.hpp"
#ifndef FLAGCV_TEST_HIP
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <chrono>
#include <functional>
#include <iomanip>
#include <string_view>

namespace {
using Submit = std::function<void()>;
#ifndef FLAGCV_TEST_HIP
using PerfClock = std::chrono::steady_clock;
using flagcv_test::cuda_check;

struct Timing {
  double event_us;
  double wall_us;
};

Timing sample(const Submit& submit, cudaStream_t stream, cudaEvent_t start, cudaEvent_t end) {
  constexpr int calls = 12;
  const auto wall_start = PerfClock::now();
  submit();
  cuda_check(cudaStreamSynchronize(stream));
  const auto wall_us = std::chrono::duration<double, std::micro>(PerfClock::now() - wall_start).count();
  cuda_check(cudaEventRecord(start, stream));
  for (int i = 0; i < calls; ++i) submit();
  cuda_check(cudaEventRecord(end, stream));
  cuda_check(cudaEventSynchronize(end));
  float ms = 0;
  cuda_check(cudaEventElapsedTime(&ms, start, end));
  return {1e3 * ms / calls, wall_us};
}

cv::cuda::GpuMat view_cuda(const Image& image) {
  const auto view = image.view();
  return {view.height(), view.width(), image.host.type(), view.data(),
          static_cast<std::size_t>(view.row_pitch())};
}

bool matches(const cv::Mat& actual, const cv::Mat& expected) {
  if (actual.type() != expected.type() || actual.size() != expected.size()) return false;
  if (actual.depth() != CV_32F) return bytes_equal(actual, expected);
  for (int y = 0; y < actual.rows; ++y) {
    const auto* a = actual.ptr<float>(y);
    const auto* b = expected.ptr<float>(y);
    for (int x = 0; x < actual.cols * actual.channels(); ++x) {
      if (a[x] == b[x] || (std::isnan(a[x]) && std::isnan(b[x]))) continue;
      if (!std::isfinite(a[x]) || !std::isfinite(b[x]) ||
          std::abs(double(a[x]) - b[x]) > std::min(1e-4, 5e-5 * std::max(1.0, std::abs(double(b[x])))))
        return false;
    }
  }
  return true;
}

void benchmark(const Stream& stream, const Image& dst, const cv::Mat& reference,
               const std::string& id, bool transparent, const Submit& candidate,
               const std::function<void(cv::cuda::GpuMat&)>& native) {
  Image baseline(dst.roi.height, dst.roi.width, dst.host.type());
  dst.pixels().copyTo(baseline.host);
  baseline.upload(stream.value);
  auto output = view_cuda(baseline);
  const auto* expected_pointer = output.data;
  const Submit compare = [&] { native(output); };
  const char* classification = "COMPARABLE";
  try {
    compare();
    stream.wait();
    if (output.data != expected_pointer || output.size() != reference.size())
      classification = "NATIVE_LAYOUT_CHANGED";
    else if (!matches(baseline.download(stream), reference))
      classification = "SEMANTICS_DIFFER";
  } catch (const cv::Exception& error) {
    classification = "NATIVE_REFUSED";
    auto message = error.err;
    for (auto& ch : message)
      if (ch == '\n' || ch == '\r') ch = ' ';
    std::cerr << id << ',' << std::quoted(message) << '\n';
  }

  const bool native_valid = std::string_view(classification) == "COMPARABLE" ||
                            std::string_view(classification) == "SEMANTICS_DIFFER";
  struct Events {
    cudaEvent_t start{}, end{};
    Events() { cuda_check(cudaEventCreate(&start)); cuda_check(cudaEventCreate(&end)); }
    ~Events() { cudaEventDestroy(start); cudaEventDestroy(end); }
  } events;
  for (int i = 0; i < 8; ++i) {
    candidate();
    if (native_valid) compare();
  }
  stream.wait();
  auto measure = [&](const Submit& call) { return sample(call, stream.value, events.start, events.end); };
  for (int i = 0; i < 11; ++i) {
    Timing flag{}, cuda{};
    if (native_valid && i % 2) cuda = measure(compare);
    flag = measure(candidate);
    if (native_valid && i % 2 == 0) cuda = measure(compare);
    std::cout << "PERF," << id << ',' << classification << ",repeat," << i << ','
              << flag.event_us << ',' << flag.wall_us << ',';
    if (native_valid) std::cout << cuda.event_us << ',' << cuda.wall_us;
    else std::cout << ',';
    std::cout << '\n';
  }
  if (transparent && !native_valid) {
    Image initial(dst.roi.height, dst.roi.width, dst.host.type());
    dst.pixels().copyTo(initial.host);
    initial.upload(stream.value);
    auto src_view = view_cuda(initial);
    auto dst_view = view_cuda(dst);
    const Submit reset_and_submit = [&] {
      cuda_check(cudaMemcpy2DAsync(dst_view.data, dst_view.step, src_view.data, src_view.step,
                                  static_cast<std::size_t>(src_view.cols) * src_view.elemSize(),
                                  src_view.rows, cudaMemcpyDeviceToDevice, stream.value));
      candidate();
    };
    for (int i = 0; i < 11; ++i) {
      const auto timing = measure(reset_and_submit);
      std::cout << "PERF," << id << ',' << classification << ",reset_included," << i << ','
                << timing.event_us << ',' << timing.wall_us << ",,\n";
    }
  }
}

#endif

struct Case {
  std::string id;
  int width, height, type, interpolation, border;
  bool flag;
  int layout;
  PerspectiveMatrix matrix;
  std::string locality;
};
const Case cases[] = {
#include "geometry_cases.inc"
};

void measure_case(flagcv::Handle& handle, const flagcv_test::Stream& stream, const Case& item) {
  const int w = item.width, h = item.height;
  const int output_padding = item.id == "perspective-4k-cubic-replicate" ? 3 : 2;
  Image src(h, w, item.type, 3, 2, 5), dst(h, w, item.type, 5, 3, output_padding);
  fill(src, dst);
  src.upload(stream.value);
  dst.upload(stream.value);
#ifndef FLAGCV_TEST_HIP
  auto cuda_stream = cv::cuda::StreamAccessor::wrapStream(stream.value);
  const auto input = view_cuda(src);
#endif
  cv::Mat ref = dst.pixels().clone();
  const auto mode = modes[item.interpolation];
  const auto border = borders[item.border];
  const cv::Scalar cv_value(value[0], value[1], value[2], value[3]);
  if (item.layout < 0) {
    auto cpu_flags = mode.cpu | (item.flag ? cv::WARP_INVERSE_MAP : 0);
    cv::Mat transform(3, 3, CV_64F, const_cast<double*>(item.matrix.data()));
    cv::warpPerspective(src.pixels(), ref, transform, ref.size(), cpu_flags, border.cpu,
                        cv_value, cv::ALGO_HINT_ACCURATE);
    const Submit candidate = [&] {
      flagcv::warpPerspective(handle, src.view(), dst.view(), item.matrix, mode.gpu,
                              border.gpu, value, item.flag, stream.value);
    };
#ifndef FLAGCV_TEST_HIP
    const auto native = [&](cv::cuda::GpuMat& output) {
      cv::cuda::warpPerspective(input, output, transform, ref.size(), cpu_flags, border.cpu,
                                cv_value, cuda_stream);
    };
#endif
    candidate();
    check(src, dst, stream, ref);
#ifdef FLAGCV_TEST_HIP
    live::candidate_only(item.id,candidate,stream.value);
#else
    if (live::verify_only()) live::verified(item.id);
    else benchmark(stream, dst, ref, item.id, item.border == 5, candidate, native);
#endif
    return;
  }

  Image first(h,w,item.layout==0?CV_32FC2:item.layout==1?CV_32FC1:CV_16SC2,2,4,3);
  Image second(h,w,item.layout==1?CV_32FC1:CV_16UC1,3,1,4);
  cv::Mat coords(h,w,CV_32FC2);
  for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
    const double u = item.locality=="scatter" ? (937*x+719*y+17)%w+.25 :
                     x+.25-(item.locality=="outside"?11:0);
    const double v = item.locality=="scatter" ? (433*x+179*y+19)%h+.5 : y+.5;
    coords.at<cv::Vec2f>(y,x) = {float(u-(item.flag?x:0)),float(v-(item.flag?y:0))};
  }
  cv::Mat map1=first.pixels(), map2=item.layout==0?cv::Mat():second.pixels();
  if (item.layout==0) coords.copyTo(map1);
  else if (item.layout==1) {
    cv::extractChannel(coords,map1,0);
    cv::extractChannel(coords,map2,1);
  } else {
    cv::convertMaps(coords,cv::noArray(),map1,map2,CV_16SC2);
  }
  first.upload(stream.value);
  if (item.layout!=0) second.upload(stream.value);
  const auto cpu_flags = mode.cpu | (item.flag ? cv::WARP_RELATIVE_MAP : 0);
  cv::remap(src.pixels(),ref,map1,map2,cpu_flags,border.cpu,cv_value,cv::ALGO_HINT_ACCURATE);
  const Submit candidate = [&] {
    flagcv::remap(handle,src.view(),dst.view(),first.view(),item.layout==0?flagcv::GpuMat{}:second.view(),
                  mode.gpu,border.gpu,value,item.flag,stream.value);
  };
#ifndef FLAGCV_TEST_HIP
  const auto cuda_first=view_cuda(first), cuda_second=item.layout==0?cv::cuda::GpuMat{}:view_cuda(second);
  const auto native = [&](cv::cuda::GpuMat& output) {
    cv::cuda::remap(input,output,cuda_first,cuda_second,cpu_flags,border.cpu,cv_value,cuda_stream);
  };
#endif
  candidate();
  check(src, dst, stream, ref);
#ifdef FLAGCV_TEST_HIP
  live::candidate_only(item.id,candidate,stream.value);
#else
  if (live::verify_only()) live::verified(item.id);
  else benchmark(stream, dst, ref, item.id, item.border == 5, candidate, native);
#endif
}
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  cv::setNumThreads(1);
  flagcv::Handle handle(FLAGCV_LIVE_BACKEND, 0);
  flagcv_test::Stream stream;
  int failures=0;
  bool found=false;
  for (const auto& item:cases) {
    if (item.id != argv[1]) continue;
    found=true;
    try { measure_case(handle,stream,item); }
    catch (const std::exception& exc) {
      ++failures;
      std::cerr<<item.id<<" CANDIDATE_OR_HARNESS_FAILED "<<exc.what()<<'\n';
    }
    std::cout<<std::flush;
  }
  return failures != 0 || !found;
}
