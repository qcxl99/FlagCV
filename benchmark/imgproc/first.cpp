#include <flagcv/flagcv.hpp>
#ifndef FLAGCV_TEST_HIP
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <opencv2/imgproc.hpp>
#include "live_device.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace flagcv;

namespace {
using Clock = std::chrono::steady_clock;
double since_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}
using Call = std::function<void()>;
constexpr int kSamples = 11;
constexpr int kBatch = 12;
struct Case {
  const char* id;
  std::string_view operator_name;
  int sw, sh, dw, dh, depth, channels, mode, code, geometry, border;
  double fx, fy;
};

const Case cases[] = {
#include "first_cases.inc"
};
constexpr AffineMatrix transforms[] = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                       {1.0, 0.0, 0.5, 0.0, 1.0, 0.5},
                                       {0.9, -0.43589, 40.0, 0.43589, 0.9, -20.0}};

#ifndef FLAGCV_TEST_HIP
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Events {
  cudaEvent_t start, finish;
  Events() { check(cudaEventCreate(&start)); check(cudaEventCreate(&finish)); }
  ~Events() { cudaEventDestroy(start); cudaEventDestroy(finish); }
};


struct Sample { double wall, batch, event; };

Sample measure(const Call& call, cudaStream_t stream, const Events& events) {
  auto start=Clock::now();
  call();
  check(cudaStreamSynchronize(stream));
  double wall=since_us(start);
  start=Clock::now();
  for (int i=0; i<kBatch; ++i) call();
  check(cudaStreamSynchronize(stream));
  double batch=since_us(start)/kBatch;
  check(cudaEventRecord(events.start, stream));
  for (int i=0; i<kBatch; ++i) call();
  check(cudaEventRecord(events.finish, stream));
  check(cudaEventSynchronize(events.finish));
  float ms=0;
  check(cudaEventElapsedTime(&ms,events.start,events.finish));
  return {wall,batch,1000*ms/kBatch};
}

#else
struct Events {};
#endif

Depth depth(int cv_depth) {
  return cv_depth==CV_8U ? Depth::k8U : cv_depth==CV_16U ? Depth::k16U : Depth::k32F;
}

#ifndef FLAGCV_TEST_HIP
cv::cuda::GpuMat comparator_view(const GpuMat& image, int type) {
  return {image.height(),image.width(),type,image.data(),static_cast<std::size_t>(image.row_pitch())};
}

void require_matched_output_layout(std::string_view id, const GpuMat& candidate, std::size_t offset,
                                  const GpuMat& native, std::size_t native_offset) {
  if (candidate.row_pitch()!=native.row_pitch() || offset!=native_offset)
    throw std::runtime_error("candidate and CUDA output layouts differ");
  std::cerr<<"LIVE_LAYOUT,"<<id<<','<<candidate.row_pitch()<<','<<offset<<','
           <<native.row_pitch()<<','<<native_offset<<'\n';
}
#endif

Border border(int cv_border) {
  return cv_border==cv::BORDER_REPLICATE ? Border::kReplicate :
         cv_border==cv::BORDER_TRANSPARENT ? Border::kTransparent : Border::kConstant;
}

struct Table {
  std::unique_ptr<flagcv_test::DeviceMemory> buffer;
  std::size_t bytes = 0;
  DeviceWorkspace view() const { return {buffer ? buffer->ptr : nullptr, bytes}; }
};

Table upload(const void* data, std::size_t bytes, const flagcv_test::Stream& stream) {
  Table table;
  if (bytes) {
    table.bytes = bytes;
    table.buffer = std::make_unique<flagcv_test::DeviceMemory>(bytes);
    flagcv_test::cuda_check(cudaMemcpyAsync(table.buffer->ptr, data, bytes,
                                           cudaMemcpyHostToDevice, stream.value));
    stream.wait();
  }
  return table;
}

void require_correct(const Case& item,const cv::Mat& src,const cv::Mat& got,const cv::Mat& expected) {
  if (expected.size()!=got.size() || expected.type()!=got.type())
    throw std::runtime_error("shape mismatch vs fixed CPU reference");
  if (item.depth!=CV_32F) {
    if (cv::norm(got,expected,cv::NORM_INF)!=0)
      throw std::runtime_error("integer mismatch vs fixed CPU reference");
    return;
  }
  for (int y=0; y<got.rows; ++y) {
    const float* a=got.ptr<float>(y);
    const float* r=expected.ptr<float>(y);
    for (int x=0; x<got.cols*got.channels(); ++x) {
      double bound=0;
      if (item.operator_name=="resize")
        bound=item.mode==0 ? 0 : item.mode==1 ? 4.948e-7 : item.mode==2 ? 1.401e-6 : 8.419e-7;
      if (item.operator_name=="warp")
        bound=std::min(1e-4,5e-5*std::max(1.,std::abs(double(r[x]))));
      if (item.operator_name=="color" && !(item.code<=5 || item.code==8 || item.code==9)) {
        const int channel=x/got.channels();
        const float* pixel=src.ptr<float>(y)+channel*src.channels();
        double scale=std::max(1.,std::abs(double(r[x])));
        for (int c=0;c<src.channels();++c) scale=std::max(scale,std::abs(double(pixel[c])));
        bound=std::min(1e-4,8*std::ldexp(1.,-24)*scale);
      }
      if (!std::isfinite(a[x]) || !std::isfinite(r[x]) || std::abs(double(a[x])-r[x])>bound)
        throw std::runtime_error("F32 mismatch vs fixed CPU reference");
    }
  }
}

void run(const Case& item, Handle& handle, const flagcv_test::Stream& stream, [[maybe_unused]] const Events* events,
         std::array<Table,5>& shared_tables) {
  const auto native=stream.value;
#ifndef FLAGCV_TEST_HIP
  auto comparator_stream = cv::cuda::StreamAccessor::wrapStream(native);
#endif
  cv::Mat input(item.sh,item.sw,CV_MAKETYPE(item.depth,item.channels));
  if (const char* path = std::getenv("FLAGCV_IMGP_INPUT")) {
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data), input.total() * input.elemSize()) ||
        file.peek() != std::ifstream::traits_type::eof())
      throw std::runtime_error("invalid shared input byte count");
  } else {
    cv::RNG rng(421);
    rng.fill(input,cv::RNG::UNIFORM,item.depth==CV_32F ? -2 : 0,
             item.depth==CV_8U ? 256 : item.depth==CV_16U ? 65536 : 2);
  }
  cv::Size size(item.dw,item.dh);
  cv::Mat cpu, cpu_direct, initial;
  cv::Mat matrix(2,3,CV_64F,const_cast<double*>(transforms[item.geometry].data()));
  live::Image src, candidate;
  Table tables;
#ifndef FLAGCV_TEST_HIP
  cv::cuda::GpuMat baseline;
#endif
  src.upload(input,stream);
  const auto input_view=src.view();
#ifndef FLAGCV_TEST_HIP
  auto native_src = comparator_view(input_view, input.type());
#endif

  double cold=0;
  Call actual;
#ifndef FLAGCV_TEST_HIP
  Call comparison;
#endif
  std::unique_ptr<ResizePlan> plan;
  const auto interpolation=static_cast<Interpolation>(item.mode);
  if (item.operator_name=="resize") {
    const auto start=Clock::now();
    plan=std::make_unique<ResizePlan>(Size{item.sw,item.sh},depth(item.depth),item.channels,
                                      ResizeOptions{{item.dw,item.dh},item.fx,item.fy,interpolation});
    const auto dsize=plan->destination_size();
    size={dsize.width,dsize.height};
    tables=upload(plan->table_data(),plan->table_bytes(),stream);
    cold=since_us(start);
    cv::resize(input,cpu_direct,{item.dw,item.dh},item.fx,item.fy,item.mode);
    if (size!=cpu_direct.size())
      throw std::runtime_error("resize size disagrees with direct fixed CPU call");
    cpu=cpu_direct;
    candidate.create(size,input.type());
    candidate.initialize(cv::Scalar::all(0),stream);
    actual=[&,opts=tables.view()] { resize(handle,input_view,candidate.view(),*plan,opts,native); };
#ifndef FLAGCV_TEST_HIP
    comparison=[&] { cv::cuda::resize(native_src,baseline,{item.dw,item.dh},item.fx,item.fy,item.mode,comparator_stream); };
#endif
  } else if (item.operator_name=="warp") {
    const int cpu_width=item.mode==cv::INTER_CUBIC ? size.width : (size.width+255)/128*128;
    const char* background = std::getenv("FLAGCV_IMGP_BACKGROUND");
    cv::Mat padded(size.height,cpu_width,input.type(),
                   background ? cv::Scalar::all(0) : cv::Scalar(19,23,33,18));
    if (background && std::string_view(background) == "input_pixels") {
      if (input.size() != size) throw std::runtime_error("input background extents differ");
      input.copyTo(padded(cv::Rect(0,0,size.width,size.height)));
    }
    initial = padded(cv::Rect(0,0,size.width,size.height)).clone();
    cv::warpAffine(input,padded,matrix,padded.size(),item.mode,item.border);
    cpu=padded(cv::Rect(0,0,size.width,size.height)).clone();
    candidate.upload(initial,stream);
    actual=[&] { warpAffine(handle,input_view,candidate.view(),transforms[item.geometry],interpolation,
                            border(item.border),{},false,native); };
#ifndef FLAGCV_TEST_HIP
    comparison=[&] { cv::cuda::warpAffine(native_src,baseline,matrix,size,item.mode,item.border,{},comparator_stream); };
#endif
  } else if (item.operator_name=="color") {
    cv::cvtColor(input,cpu,item.code);
    size=cpu.size();
    const auto family=cvtColorTableKind(item.code);
    auto& family_table=shared_tables[static_cast<std::size_t>(family)];
    if (family!=ColorTables::kNone && !family_table.buffer) {
      const auto start=Clock::now();
      const auto host=cvtColorTableData(family);
      family_table=upload(host.data,host.bytes,stream);
      cold=since_us(start);
    }
    candidate.create(size,cpu.type());
    candidate.initialize(cv::Scalar::all(0),stream);
    actual=[&,opts=family_table.view()] { cvtColor(handle,input_view,candidate.view(),item.code,opts,native); };
#ifndef FLAGCV_TEST_HIP
    comparison=[&] { cv::cuda::cvtColor(native_src,baseline,item.code,cpu.channels(),comparator_stream); };
#endif
  } else {
    throw std::invalid_argument("unknown first-family operation");
  }

  actual();
  cv::Mat got=candidate.download(stream);
  require_correct(item,input,got,cpu);
  if (!flagcv_test::bytes_equal(src.download(stream), input))
    throw std::runtime_error("source modified");
#ifdef FLAGCV_TEST_HIP
  (void)cold;
  live::candidate_only(item.id, actual, native);
#else
  if (live::verify_only()) { live::verified(item.id); return; }
  live::Image baseline_storage;
  if (initial.empty()) {
    baseline_storage.create(size,cpu.type());
    baseline_storage.initialize(cv::Scalar::all(0),stream);
  } else {
    baseline_storage.upload(initial,stream);
  }
  baseline=comparator_view(baseline_storage.view(),cpu.type());
  const cv::Rect roi(0,0,size.width,size.height);
  require_matched_output_layout(item.id,candidate.view(),candidate.offset(roi),
                                baseline_storage.view(),baseline_storage.offset(roi));
  const bool direct_equal=cpu_direct.empty() || cv::norm(cpu,cpu_direct,cv::NORM_INF)==0;
  double native_delta=0;
  bool comparable=true;
  try {
    const auto saved=baseline.data;
    comparison();
    stream.wait();
    if (baseline.data!=saved || baseline.size()!=size || baseline.type()!=cpu.type() ||
        baseline.step!=static_cast<std::size_t>(baseline_storage.view().row_pitch()))
      throw cv::Exception(cv::Error::StsUnmatchedFormats,"native produced a different output layout",
                          "paired-cpp",__FILE__,__LINE__);
    const auto native_result=baseline_storage.download(stream);
    if (!flagcv_test::bytes_equal(src.download(stream),input))
      throw std::runtime_error("source modified by CUDA comparator");
    native_delta=cv::norm(native_result,cpu,cv::NORM_INF);
    try { require_correct(item,input,native_result,cpu); }
    catch (const std::runtime_error&) { comparable=false; }
  } catch(const cv::Exception& error) {
    std::cout<<item.id<<",BASELINE_REFUSED,"<<std::quoted(error.err)<<"\n";
    for (int i=0; i<kSamples; ++i) {
      const auto f=measure(actual,native,*events);
      std::cout<<item.id<<",CANDIDATE_ONLY,"<<i<<",F,"<<f.wall<<",,"<<f.batch<<",,"<<f.event<<",,,,\n";
    }
    return;
  }
  for (int i=0; i<20; ++i) { actual(); comparison(); }
  stream.wait();
  for (int i=0; i<kSamples; ++i) {
    Sample f,c;
    if (i%2) { c=measure(comparison,native,*events); f=measure(actual,native,*events); }
    else { f=measure(actual,native,*events); c=measure(comparison,native,*events); }
    std::cout<<item.id<<(comparable?",COMPARABLE,":",SEMANTICS_DIFFER,")<<i<<','<<(i%2?"CF":"FC")<<','<<f.wall<<','<<c.wall<<','
             <<f.batch<<','<<c.batch<<','<<f.event<<','<<c.event<<','<<native_delta<<','
             <<cold<<','<<direct_equal<<'\n';
  }
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 15) return 2;
  Handle handle(FLAGCV_LIVE_BACKEND, 0);
  flagcv_test::Stream stream;
  const auto events = live::verify_only() ? nullptr : std::make_unique<Events>();
  cv::setNumThreads(1);
  std::array<Table,5> shared_tables;
  std::cout << "id,status,sample,order,flagcv_wall_us,cuda_wall_us,flagcv_batch_wall_us,"
               "cuda_batch_wall_us,flagcv_event_us,cuda_event_us,cuda_cpu_max_abs,cold_table_us,"
               "portable_reference_equals_direct_cpu\n";
  int errors=0;
  bool found=false;
  std::vector<Case> selected;
  if (argc == 15) {
    selected.push_back({argv[1],argv[2],std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),
        std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10]),
        std::stoi(argv[11]),std::stoi(argv[12]),std::stod(argv[13]),std::stod(argv[14])});
    if (selected[0].geometry < 0 || selected[0].geometry >= 3) return 2;
  } else {
    selected.assign(std::begin(cases), std::end(cases));
  }
  for (const auto& item:selected) {
    if (std::string_view(argv[1]) != item.id) continue;
    found=true;
    try {
      run(item,handle,stream,events.get(),shared_tables);
    } catch(const std::exception& e) {
      std::string message=e.what();
      for (auto& ch:message) if (ch=='\n'||ch=='\r') ch=' ';
      std::cout<<item.id<<",FLAGCV_FAIL,"<<std::quoted(message)<<"\n";
      ++errors;
    }
    std::cout<<std::flush;
  }
  return errors || !found ? 1 : 0;
}
