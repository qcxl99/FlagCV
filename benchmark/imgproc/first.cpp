#include <flagcv/flagcv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

using namespace flagcv;

namespace {
using Clock = std::chrono::steady_clock;
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

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Events {
  cudaEvent_t start, finish;
  Events() { check(cudaEventCreate(&start)); check(cudaEventCreate(&finish)); }
  ~Events() { cudaEventDestroy(start); cudaEventDestroy(finish); }
};

double since_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}

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

Depth depth(int cv_depth) {
  return cv_depth==CV_8U ? Depth::k8U : cv_depth==CV_16U ? Depth::k16U : Depth::k32F;
}

GpuMat view(const cv::cuda::GpuMat& mat) {
  return {mat.data,mat.cols,mat.rows,depth(mat.depth()),mat.channels(),static_cast<std::int64_t>(mat.step)};
}

Border border(int cv_border) {
  return cv_border==cv::BORDER_REPLICATE ? Border::kReplicate :
         cv_border==cv::BORDER_TRANSPARENT ? Border::kTransparent : Border::kConstant;
}

cv::cuda::GpuMat upload(const void* data, std::size_t bytes, cv::cuda::Stream& stream) {
  cv::cuda::GpuMat out;
  if (bytes) {
    const cv::Mat host(1,static_cast<int>(bytes),CV_8UC1,const_cast<void*>(data));
    out.upload(host,stream);
    stream.waitForCompletion();
  }
  return out;
}

DeviceWorkspace workspace(const cv::cuda::GpuMat& mat) {
  return {mat.data,static_cast<std::size_t>(mat.cols)};
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

void run(const Case& item, Handle& handle, cv::cuda::Stream& stream, const Events& events,
         std::array<cv::cuda::GpuMat,5>& shared_tables) {
  const auto native=cv::cuda::StreamAccessor::getStream(stream);
  cv::Mat input(item.sh,item.sw,CV_MAKETYPE(item.depth,item.channels));
  cv::RNG rng(421);
  rng.fill(input,cv::RNG::UNIFORM,item.depth==CV_32F ? -2 : 0,
           item.depth==CV_8U ? 256 : item.depth==CV_16U ? 65536 : 2);
  cv::Size size(item.dw,item.dh);
  cv::Mat cpu, cpu_direct;
  cv::Mat matrix(2,3,CV_64F,const_cast<double*>(transforms[item.geometry].data()));
  cv::cuda::GpuMat src, candidate, baseline, tables;
  src.upload(input,stream);
  const auto input_view=view(src);

  double cold=0;
  Call actual, comparison;
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
    baseline.create(size,input.type());
    actual=[&,opts=workspace(tables)] { resize(handle,input_view,view(candidate),*plan,opts,native); };
    comparison=[&] { cv::cuda::resize(src,baseline,{item.dw,item.dh},item.fx,item.fy,item.mode,stream); };
  } else if (item.operator_name=="warp") {
    const int cpu_width=item.mode==cv::INTER_CUBIC ? size.width : (size.width+255)/128*128;
    cv::Mat padded(size.height,cpu_width,input.type(),cv::Scalar(19,23,33,18));
    cv::warpAffine(input,padded,matrix,padded.size(),item.mode,item.border);
    cpu=padded(cv::Rect(0,0,size.width,size.height)).clone();
    candidate.create(size,input.type());
    candidate.setTo(cv::Scalar(19,23,33,18),stream);
    baseline.create(size,input.type());
    actual=[&] { warpAffine(handle,input_view,view(candidate),transforms[item.geometry],interpolation,
                            border(item.border),{},false,native); };
    comparison=[&] { cv::cuda::warpAffine(src,baseline,matrix,size,item.mode,item.border,{},stream); };
  } else {
    cv::cvtColor(input,cpu,item.code);
    size=cpu.size();
    const auto family=cvtColorTableKind(item.code);
    auto& family_table=shared_tables[static_cast<std::size_t>(family)];
    if (family!=ColorTables::kNone && family_table.empty()) {
      const auto start=Clock::now();
      const auto host=cvtColorTableData(family);
      family_table=upload(host.data,host.bytes,stream);
      cold=since_us(start);
    }
    candidate.create(size,cpu.type());
    baseline.create(size,cpu.type());
    actual=[&,opts=workspace(family_table)] { cvtColor(handle,input_view,view(candidate),item.code,opts,native); };
    comparison=[&] { cv::cuda::cvtColor(src,baseline,item.code,cpu.channels(),stream); };
  }

  actual();
  stream.waitForCompletion();
  cv::Mat got;
  candidate.download(got);
  require_correct(item,input,got,cpu);
  const bool direct_equal=cpu_direct.empty() || cv::norm(cpu,cpu_direct,cv::NORM_INF)==0;
  double native_delta=0;
  bool comparable=true;
  try {
    const auto saved=baseline.data;
    comparison();
    stream.waitForCompletion();
    if (baseline.data!=saved || baseline.size()!=size || baseline.type()!=cpu.type())
      throw cv::Exception(cv::Error::StsUnmatchedFormats,"native produced a different output layout",
                          "paired-cpp",__FILE__,__LINE__);
    cv::Mat native_result;
    baseline.download(native_result);
    native_delta=cv::norm(native_result,cpu,cv::NORM_INF);
    try { require_correct(item,input,native_result,cpu); }
    catch (const std::runtime_error&) { comparable=false; }
  } catch(const cv::Exception& error) {
    std::cout<<item.id<<",BASELINE_REFUSED,"<<std::quoted(error.err)<<"\n";
    for (int i=0; i<kSamples; ++i) {
      const auto f=measure(actual,native,events);
      std::cout<<item.id<<",CANDIDATE_ONLY,"<<i<<",F,"<<f.wall<<",,"<<f.batch<<",,"<<f.event<<",,,,\n";
    }
    return;
  }
  for (int i=0; i<20; ++i) { actual(); comparison(); }
  stream.waitForCompletion();
  for (int i=0; i<kSamples; ++i) {
    Sample f,c;
    if (i%2) { c=measure(comparison,native,events); f=measure(actual,native,events); }
    else { f=measure(actual,native,events); c=measure(comparison,native,events); }
    std::cout<<item.id<<(comparable?",COMPARABLE,":",SEMANTICS_DIFFER,")<<i<<','<<(i%2?"CF":"FC")<<','<<f.wall<<','<<c.wall<<','
             <<f.batch<<','<<c.batch<<','<<f.event<<','<<c.event<<','<<native_delta<<','
             <<cold<<','<<direct_equal<<'\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Handle handle;
  cv::cuda::Stream stream(cudaStreamNonBlocking);
  Events events;
  cv::setNumThreads(1);
  std::array<cv::cuda::GpuMat,5> shared_tables;
  std::cout << "id,status,sample,order,flagcv_wall_us,cuda_wall_us,flagcv_batch_wall_us,"
               "cuda_batch_wall_us,flagcv_event_us,cuda_event_us,cuda_cpu_max_abs,cold_table_us,"
               "portable_reference_equals_direct_cpu\n";
  int errors=0;
  bool found=false;
  for (const auto& item:cases) {
    if (std::string_view(argv[1]) != item.id) continue;
    found=true;
    try {
      run(item,handle,stream,events,shared_tables);
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
