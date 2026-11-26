#include <flagcv/flagcv.hpp>
#ifndef FLAGCV_TEST_HIP
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "live_device.hpp"

namespace {
using flagcv_test::Image;
using flagcv_test::DeviceMemory;
using flagcv_test::bytes_equal;
using flagcv_test::cuda_check;
using NativeStream = flagcv_test::Stream;
using Clock = std::chrono::steady_clock;
using Call = std::function<double()>;
const double no_scalar = std::numeric_limits<double>::quiet_NaN();

const std::pair<int, flagcv::Border> borders[] = {
    {cv::BORDER_CONSTANT, flagcv::Border::kConstant},
    {cv::BORDER_REPLICATE, flagcv::Border::kReplicate},
    {cv::BORDER_REFLECT, flagcv::Border::kReflect},
    {cv::BORDER_REFLECT_101, flagcv::Border::kReflect101},
    {cv::BORDER_WRAP, flagcv::Border::kWrap},
};

#ifndef FLAGCV_TEST_HIP
cv::cuda::GpuMat external(const Image& image) {
  auto view = image.view();
  return {view.height(), view.width(), image.host.type(), view.data(),
          static_cast<std::size_t>(view.row_pitch())};
}

#endif

void initialize(Image& image, cv::RNG& rng) {
  cv::Mat buffer(image.host.size(), CV_MAKETYPE(CV_32F, image.host.channels()));
  const bool integer = image.host.depth() <= CV_32S;
  const bool unsigned_input = image.host.depth() == CV_8U || image.host.depth() == CV_16U;
  rng.fill(buffer, cv::RNG::UNIFORM, unsigned_input ? 0 : (integer ? -127 : -2), integer ? 255 : 2);
  buffer.convertTo(image.host, image.host.depth());
}

bool matches(const cv::Mat& actual, const cv::Mat& oracle, double tolerance) {
  if (actual.size() != oracle.size() || actual.type() != oracle.type()) return false;
  if (!tolerance) return bytes_equal(actual, oracle);
  cv::Mat a, b;
  actual.convertTo(a, CV_64F);
  oracle.convertTo(b, CV_64F);
  for (int y = 0; y < a.rows; ++y) {
    const auto* lhs = a.ptr<double>(y);
    const auto* rhs = b.ptr<double>(y);
    for (int x = 0; x < a.cols * a.channels(); ++x) {
      if (lhs[x] == rhs[x] || (std::isnan(lhs[x]) && std::isnan(rhs[x]))) continue;
      if (!std::isfinite(lhs[x]) || !std::isfinite(rhs[x]) ||
          std::abs(lhs[x] - rhs[x]) > tolerance * std::max(1.0, std::abs(rhs[x])))
        return false;
    }
  }
  return true;
}

void validate_candidate(const std::string& id, const Image& src, const Image& dst,
                        const cv::Mat& expected, const NativeStream& stream, const Call& candidate,
                        double tolerance, double scalar) {
  const auto got_scalar = candidate();
  auto received = dst.download(stream);
  if ((!std::isnan(scalar) && got_scalar != scalar) || !matches(received(dst.roi), expected, tolerance))
    throw std::runtime_error(id + ": CANDIDATE_CPU_DIFFER");
  dst.pixels().copyTo(received(dst.roi));
  if (!bytes_equal(received, dst.host))
    throw std::runtime_error(id + ": destination guard modified");
  if (!bytes_equal(src.download(stream), src.host))
    throw std::runtime_error(id + ": source bytes modified");
}

#ifndef FLAGCV_TEST_HIP
struct Events {
  cudaEvent_t start{}, end{};
  Events() { cuda_check(cudaEventCreate(&start)); cuda_check(cudaEventCreate(&end)); }
  ~Events() { cudaEventDestroy(start); cudaEventDestroy(end); }
};

struct Times { double wall{}, event{}; };

Times measure(const Call& work, const NativeStream& stream, const Events& events, bool host_result) {
  const auto start = Clock::now();
  work();
  stream.wait();
  Times out{std::chrono::duration<double, std::micro>(Clock::now() - start).count(), 0};
  if (!host_result) {
    constexpr int calls = 8;
    cuda_check(cudaEventRecord(events.start, stream.value));
    for (int i = 0; i < calls; ++i) work();
    cuda_check(cudaEventRecord(events.end, stream.value));
    cuda_check(cudaEventSynchronize(events.end));
    float ms{};
    cuda_check(cudaEventElapsedTime(&ms, events.start, events.end));
    out.event = 1000 * ms / calls;
  }
  return out;
}

void timing(const std::string& id, const Image& src, const Image& dst, const Image& baseline,
            const cv::Mat& expected, const NativeStream& stream, const Events* events,
            const Call& candidate, const Call& other, cv::cuda::GpuMat& output,
            double tolerance, double scalar = no_scalar, bool host_result = false) {
  validate_candidate(id, src, dst, expected, stream, candidate, tolerance, scalar);
  if (live::verify_only()) { live::verified(id); return; }
  std::string category = "COMPARABLE";
  bool native_admitted = true;
  const auto* pointer = output.data;
  try {
    const auto scalar_cuda = other();
    stream.wait();
    if (output.data != pointer || output.type() != expected.type() || output.size() != expected.size()) {
      category = "NATIVE_LAYOUT_CHANGED";
      native_admitted = false;
    } else if ((!std::isnan(scalar) && scalar_cuda != scalar) ||
               !matches(baseline.download(stream)(baseline.roi), expected, tolerance)) {
      category = "SEMANTICS_DIFFER";
    }
  } catch (const cv::Exception& exc) {
    category = "NATIVE_REFUSED";
    native_admitted = false;
    std::cerr << id << ',' << std::quoted(exc.err) << '\n';
  }
  for (int i = 0; i < 3; ++i) {
    candidate();
    if (native_admitted) other();
  }
  stream.wait();
  for (int i = 0; i < 11; ++i) {
    Times flag{}, native{};
    if (native_admitted && i % 2) native = measure(other, stream, *events, host_result);
    flag = measure(candidate, stream, *events, host_result);
    if (native_admitted && i % 2 == 0) native = measure(other, stream, *events, host_result);
    std::cout << id << ',' << category << ',' << i << ',' << flag.wall << ',';
    if (!host_result) std::cout << flag.event;
    std::cout << ',';
    if (native_admitted) std::cout << native.wall;
    std::cout << ',';
    if (native_admitted && !host_result) std::cout << native.event;
    std::cout << '\n';
  }
}

#else
void timing(const std::string& id, const Image& src, const Image& dst, const cv::Mat& expected, const NativeStream& stream,
            const Call& candidate, double tolerance, double scalar = no_scalar, bool host_result = false) {
  validate_candidate(id, src, dst, expected, stream, candidate, tolerance, scalar);
  live::candidate_only(id, candidate, stream.value, host_result);
}
#endif

struct Table {
  DeviceMemory memory;
  std::size_t bytes;
  Table(flagcv::HostTableData host, const NativeStream& stream) : memory(host.bytes), bytes(host.bytes) {
    cuda_check(cudaMemcpyAsync(memory.ptr, host.data, bytes, cudaMemcpyHostToDevice, stream.value));
  }
  flagcv::DeviceWorkspace view() const { return {memory.ptr, bytes}; }
};

struct Arguments {
  flagcv::Handle& handle;
  NativeStream& stream;
#ifndef FLAGCV_TEST_HIP
  cv::cuda::Stream& native;
  const Events* events;
#endif
};

void copy_sample(Arguments ctx, int width, int height, int depth, int channels, int border) {
  constexpr flagcv::BorderExtents pad{3, 5, 7, 9};
  const int type = CV_MAKETYPE(depth, channels);
  Image src(height, width, type, 2, 1, 5), dst(height + 8, width + 16, type, 1, 3, 4);
  dst.host.setTo(cv::Scalar::all(91));
  dst.upload(ctx.stream.value);
#ifndef FLAGCV_TEST_HIP
  Image baseline(height + 8, width + 16, type, 1, 3, 4);
#endif
  cv::RNG rng(872);
  initialize(src, rng);
  src.upload(ctx.stream.value);
  const auto [cpu_border, flag_border] = borders[border];
  cv::Mat expected;
  const cv::Scalar constant(2.5, 3.5, 11, 15);
  cv::copyMakeBorder(src.pixels(), expected, pad.top, pad.bottom, pad.left, pad.right,
                     cpu_border | cv::BORDER_ISOLATED, constant);
#ifndef FLAGCV_TEST_HIP
  auto input = external(src), output = external(baseline);
#endif
  const auto candidate = [&] {
    flagcv::copyMakeBorder(ctx.handle, src.view(), dst.view(), pad, flag_border,
                           {2.5, 3.5, 11, 15}, nullptr, ctx.stream.value);
    return no_scalar;
  };
#ifndef FLAGCV_TEST_HIP
  const auto native = [&] {
    cv::cuda::copyMakeBorder(input, output, pad.top, pad.bottom, pad.left, pad.right,
                            cpu_border, constant, ctx.native);
    return no_scalar;
  };
#endif
  const auto id = "copy-" + std::to_string(width) + "-" + std::to_string(height) + "-" +
                  std::to_string(type) + "-" + std::to_string(border);
#ifdef FLAGCV_TEST_HIP
  timing(id, src, dst, expected, ctx.stream, candidate, 0);
#else
  timing(id, src, dst, baseline, expected, ctx.stream, ctx.events, candidate, native, output, 0);
#endif
}

void threshold_sample(Arguments ctx, int width, int height, int depth, int channels, int mode, int automatic) {
  const int type = CV_MAKETYPE(depth, channels);
  Image src(height, width, type, 2, 1, 5), dst(height, width, type, 1, 3, 4);
  dst.host.setTo(cv::Scalar::all(91));
  dst.upload(ctx.stream.value);
#ifndef FLAGCV_TEST_HIP
  Image baseline(height, width, type, 1, 3, 4);
#endif
  cv::RNG rng(876);
  initialize(src, rng);
  src.upload(ctx.stream.value);
  const auto method = automatic == cv::THRESH_TRIANGLE ? flagcv::AutomaticThreshold::kTriangle : flagcv::AutomaticThreshold::kOtsu;
  const double cutoff = automatic ? 0 : 51.5;
  const double maxval = 201.5;
  cv::Mat expected;
  const double returned = cv::threshold(src.pixels(), expected, cutoff, maxval, mode | automatic);
#ifndef FLAGCV_TEST_HIP
  auto input = external(src), output = external(baseline);
#endif
  std::unique_ptr<DeviceMemory> scratch;
  std::size_t bytes = 0;
  if (automatic) {
    bytes = flagcv::thresholdWorkspaceBytes(flagcv_test::depth_for(depth), method);
    scratch = std::make_unique<DeviceMemory>(bytes);
  }
  const auto candidate = [&] {
    const auto route = static_cast<flagcv::ThresholdMode>(mode);
    if (automatic)
      return flagcv::thresholdAutoHost(ctx.handle, src.view(), dst.view(), maxval, route, method,
                                      {scratch->ptr, bytes}, false, ctx.stream.value);
    return flagcv::threshold(ctx.handle, src.view(), dst.view(), cutoff, maxval, route, false, ctx.stream.value);
  };
#ifndef FLAGCV_TEST_HIP
  const auto native = [&] {
    return cv::cuda::threshold(input, output, cutoff, maxval, mode | automatic, ctx.native);
  };
#endif
  const auto id = "threshold-" + std::to_string(width) + "-" + std::to_string(height) + "-" +
                  std::to_string(type) + "-" + std::to_string(mode) + "-" + std::to_string(automatic);
#ifdef FLAGCV_TEST_HIP
  timing(id, src, dst, expected, ctx.stream, candidate, 0, returned, automatic != 0);
#else
  timing(id, src, dst, baseline, expected, ctx.stream, ctx.events, candidate, native, output, 0, returned, automatic != 0);
#endif
}

void spatial_sample(Arguments ctx, int width, int height, int depth, int channels, int kernel,
                    int border, bool gaussian, bool two_pass) {
  const int type = CV_MAKETYPE(depth, channels);
  Image src(height, width, type, 2, 1, 5), dst(height, width, type, 1, 3, 4);
  dst.host.setTo(cv::Scalar::all(91));
  dst.upload(ctx.stream.value);
#ifndef FLAGCV_TEST_HIP
  Image baseline(height, width, type, 1, 3, 4);
#endif
  cv::RNG rng(879);
  initialize(src, rng);
  src.upload(ctx.stream.value);
  const auto dtype = flagcv_test::depth_for(depth);
  const auto [cpu_border, flag_border] = borders[border];
#ifndef FLAGCV_TEST_HIP
  auto input = external(src), output = external(baseline);
#endif
  const auto id = std::string(gaussian ? "gauss" : "filter") + "-" + std::to_string(width) + "-" +
                  std::to_string(height) + "-" + std::to_string(type) + "-" + std::to_string(kernel) +
                  "-" + std::to_string(border) + "-" + std::to_string(two_pass);
  const double tolerance = depth == CV_32F ? 2e-5 : depth == CV_64F ? 4e-12 : 0;
#ifndef FLAGCV_TEST_HIP
  cv::Ptr<cv::cuda::Filter> native_plan;
#endif
  cv::Mat expected;
  if (gaussian) {
    const flagcv::GaussianPlan plan({width, height}, dtype, {{kernel, kernel}, 0, 0, flag_border});
    const Table table(plan.table(), ctx.stream);
    std::unique_ptr<Image> intermediate;
    if (two_pass)
      intermediate = std::make_unique<Image>(height, width,
          CV_MAKETYPE(depth == CV_8U || depth == CV_16U ? CV_32S : depth, channels));
    cv::GaussianBlur(src.pixels(), expected, {kernel, kernel}, 0, 0, cpu_border | cv::BORDER_ISOLATED);
    const auto candidate = [&] {
      if (intermediate)
        flagcv::GaussianBlur(ctx.handle, src.view(), dst.view(), plan, table.view(),
                             intermediate->view(), ctx.stream.value);
      else
        flagcv::GaussianBlur(ctx.handle, src.view(), dst.view(), plan, table.view(), ctx.stream.value);
      return no_scalar;
    };
#ifndef FLAGCV_TEST_HIP
    const auto native = [&] {
      if (!native_plan)
        native_plan = cv::cuda::createGaussianFilter(type, type, {kernel, kernel}, 0, 0, cpu_border, cpu_border);
      native_plan->apply(input, output, ctx.native);
      return no_scalar;
    };
#endif
#ifdef FLAGCV_TEST_HIP
    timing(id, src, dst, expected, ctx.stream, candidate, tolerance);
#else
    timing(id, src, dst, baseline, expected, ctx.stream, ctx.events, candidate, native, output, tolerance);
#endif
  } else {
    cv::Mat coefficients(kernel, kernel, CV_64F);
    rng.fill(coefficients, cv::RNG::UNIFORM, -.3, .3);
    const flagcv::Filter2DPlan plan({kernel, kernel}, coefficients.ptr<double>(), dtype,
                                  {{}, {-1, -1}, 0, flag_border});
    const Table table(plan.table(), ctx.stream);
    cv::filter2D(src.pixels(), expected, -1, coefficients, {-1, -1}, 0, cpu_border | cv::BORDER_ISOLATED);
    const auto candidate = [&] {
      flagcv::filter2D(ctx.handle, src.view(), dst.view(), plan, table.view(), ctx.stream.value);
      return no_scalar;
    };
#ifndef FLAGCV_TEST_HIP
    const auto native = [&] {
      if (!native_plan)
        native_plan = cv::cuda::createLinearFilter(type, type, coefficients, {-1, -1}, cpu_border);
      native_plan->apply(input, output, ctx.native);
      return no_scalar;
    };
#endif
#ifdef FLAGCV_TEST_HIP
    timing(id, src, dst, expected, ctx.stream, candidate, tolerance);
#else
    timing(id, src, dst, baseline, expected, ctx.stream, ctx.events, candidate, native, output, tolerance);
#endif
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  cv::setNumThreads(1);
  flagcv::Handle handle(FLAGCV_LIVE_BACKEND, 0);
  NativeStream stream;
#ifdef FLAGCV_TEST_HIP
  Arguments ctx{handle, stream};
#else
  auto native = cv::cuda::StreamAccessor::wrapStream(stream.value);
  const auto events = live::verify_only() ? nullptr : std::make_unique<Events>();
  Arguments ctx{handle, stream, native, events.get()};
#endif
  std::cout << "case,classification,sample,flagcv_sync_wall_us,flagcv_event_us,native_sync_wall_us,native_event_us\n";
  const std::string id = argv[1];
  std::string kind;
  std::vector<int> fields;
  const auto delimiter = id.find('-');
  if (delimiter == std::string::npos) return 2;
  kind = id.substr(0, delimiter);
  for (std::size_t start = delimiter + 1; start < id.size();) {
    const auto stop = id.find('-', start);
    fields.push_back(std::stoi(id.substr(start, stop == std::string::npos ? stop : stop-start)));
    if (stop == std::string::npos) break;
    start = stop+1;
  }
  if (fields.size() < 4) return 2;
  const int w=fields[0], h=fields[1], type=fields[2], depth=CV_MAT_DEPTH(type), cn=CV_MAT_CN(type);
  if (kind=="copy" && fields.size()==4) {
    copy_sample(ctx,w,h,depth,cn,fields[3]);
  } else if (kind=="threshold" && fields.size()==5) {
    threshold_sample(ctx,w,h,depth,cn,fields[3],fields[4]);
  } else if ((kind=="gauss" || kind=="filter") && fields.size()==6) {
    spatial_sample(ctx,w,h,depth,cn,fields[3],fields[4],kind=="gauss",fields[5]!=0);
  } else {
    return 2;
  }
}
