#include <flagcv/flagcv.hpp>
#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace {
using namespace flagcv;
thread_local std::string error, answer;
struct Prepared {
  Depth input, output;
  Size size;
  std::unique_ptr<GaussianPlan> gaussian;
  std::unique_ptr<Filter2DPlan> filter;
};

Depth from_cv(int type) {
  switch(type) {
    case 0: return Depth::k8U;
    case 2: return Depth::k16U;
    case 3: return Depth::k16S;
    case 5: return Depth::k32F;
    case 6: return Depth::k64F;
  }
  throw std::invalid_argument("unsupported spatial input depth");
}

const char* kind(Depth type) {
  switch(type) {
    case Depth::k8U: return "8U";
    case Depth::k16U: return "16U";
    case Depth::k16S: return "16S";
    case Depth::k32F: return "32F";
    case Depth::k32S: return "32S";
    case Depth::k64F: return "64F";
    default: throw std::invalid_argument("unsupported spatial pointer depth");
  }
}

Border border(int type) {
  switch(type) {
    case 0: return Border::kConstant;
    case 1: return Border::kReplicate;
    case 2: return Border::kReflect;
    case 4: return Border::kReflect101;
  }
  throw std::invalid_argument("unsupported spatial border");
}

HostTableData table(const Prepared& p) {
  return p.gaussian ? p.gaussian->table() : p.filter->table();
}

void emit(std::ostringstream& s, const LoweredOp& op) {
  s<<"{\"kernel\":\""<<op.kernel<<"\",\"images\":[";
  bool first=true;
  for (const auto& image:op.images) {
    if (!first) s<<',';
    first=false;
    s<<"{\"name\":\""<<image.name<<"\",\"address\":"<<image.address
     <<",\"depth\":\""<<kind(image.depth)<<"\",\"channels\":"<<image.channels<<'}';
  }
  s<<"],\"params\":[";
  first=true;
  for (const auto& p:op.params) {
    if (!first) s<<',';
    first=false;
    s<<"{\"name\":\""<<p.name<<"\",\"type\":\""<<
      (p.compile_time?"constexpr":p.is_fp32?"fp32":"int")<<"\",\"value\":";
    if (p.is_fp32) s<<p.fp32_value;
    else s<<p.value;
    s<<'}';
  }
  s<<"]}";
}
}  // namespace

extern "C" void* ready_gaussian(int w,int h,int depth,int kw,int kh,double sx,double sy,int edge) {
  try {
    auto result=std::make_unique<Prepared>();
    result->input=result->output=from_cv(depth);
    result->size={w,h};
    result->gaussian=std::make_unique<flagcv::GaussianPlan>(result->size,result->input,
                                       flagcv::GaussianOptions{{kw,kh},sx,sy,border(edge)});
    return result.release();
  } catch(const std::exception& e) {
    error=e.what();
    return nullptr;
  }
}

extern "C" void* ready_filter(int w,int h,int depth,int dest,int kw,int kh,const double* weights,
                               int anchor_x,int anchor_y,double delta,int edge) {
  try {
    auto result=std::make_unique<Prepared>();
    result->input=from_cv(depth);
    result->output=dest==-1 ? result->input : from_cv(dest);
    result->size={w,h};
    result->filter=std::make_unique<flagcv::Filter2DPlan>(
        flagcv::Size{kw,kh},weights,result->input,
        flagcv::Filter2DOptions{result->output,{anchor_x,anchor_y},delta,border(edge)});
    return result.release();
  } catch(const std::exception& e) {
    error=e.what();
    return nullptr;
  }
}

extern "C" std::size_t table_bytes(void* p) { return table(*static_cast<Prepared*>(p)).bytes; }
extern "C" const void* table_data(void* p) { return table(*static_cast<Prepared*>(p)).data; }
extern "C" const char* last_message() { return error.c_str(); }
extern "C" void release(void* p) { delete static_cast<Prepared*>(p); }

extern "C" const char* describe(void* p, void* i, long long ip, void* o, long long op,
                                int cn, const void* coefficients, void* scratch, long long scratch_pitch) {
  using namespace flagcv;
  try {
    const auto& plan=*static_cast<Prepared*>(p);
    const auto shape=plan.size;
    const GpuMat source(i,shape.width,shape.height,plan.input,cn,ip);
    const GpuMat target(o,shape.width,shape.height,plan.output,cn,op);
    const DeviceWorkspace data{coefficients,table(plan).bytes};
    std::ostringstream s;
    s<<std::setprecision(std::numeric_limits<float>::max_digits10)<<'[';
    if (!plan.gaussian) emit(s,lower_filter2d(source,target,*plan.filter,data));
    else if (!scratch) emit(s,lower_gaussian(source,target,*plan.gaussian,data));
    else {
      const GpuMat tmp(scratch,shape.width,shape.height,plan.gaussian->scratch_depth(),cn,scratch_pitch);
      const auto passes=lower_gaussian_separable(source,target,*plan.gaussian,data,tmp);
      emit(s,passes[0]);
      s<<',';
      emit(s,passes[1]);
    }
    s<<']';
    answer=s.str();
    return answer.c_str();
  } catch(const std::exception& e) {
    error=e.what();
    return nullptr;
  }
}
