#include <flagcv/flagcv.hpp>
#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace {
using namespace flagcv;
using Ptr = std::shared_ptr<ResizePlan>;
struct State {
  int operation, code, mode, border, depth, channels, sw, sh;
  AffineMatrix matrix;
  Ptr resize;
  HostTableData host;
};

thread_local std::string message, answer;

Depth depth_for(int depth) {
  return depth == 0 ? Depth::k8U : depth == 2 ? Depth::k16U : Depth::k32F;
}

const char* depth_name(Depth depth) {
  switch (depth) {
    case Depth::k8U: return "8U";
    case Depth::k16U: return "16U";
    case Depth::k16S: return "16S";
    case Depth::k32F: return "32F";
    case Depth::k32S: return "32S";
    default: throw std::runtime_error("unexpected bridge depth");
  }
}
}

extern "C" void* prepare(int op, int sw, int sh, int dw, int dh,
                          int dep, int cn, int mode, int code, int geometry, int border,
                          double fx, double fy) {
  using namespace flagcv;
  try {
    constexpr AffineMatrix matrices[] = {{1,0,0,0,1,0},{1,0,.5,0,1,.5},
                                        {.9,-.43589,40,.43589,.9,-20}};
    auto state=std::make_unique<State>();
    *state={op,code,mode,border,dep,cn,sw,sh,matrices[geometry],{}, {nullptr,0}};
    if (op==0) {
      state->resize=std::make_shared<ResizePlan>(Size{sw,sh},depth_for(dep),cn,
                                   ResizeOptions{{dw,dh},fx,fy,static_cast<Interpolation>(mode)});
      state->host={state->resize->table_data(),state->resize->table_bytes()};
    }
    if (op==2) state->host=cvtColorTableData(cvtColorTableKind(code));
    return state.release();
  } catch(const std::exception& e) {
    message=e.what();
    return nullptr;
  }
}

extern "C" const void* host_data(void* state) { return static_cast<State*>(state)->host.data; }
extern "C" std::size_t host_bytes(void* state) { return static_cast<State*>(state)->host.bytes; }
extern "C" void release(void* state) { delete static_cast<State*>(state); }
extern "C" const char* last_message() { return message.c_str(); }

static const char* describe_plan(void* raw, void* source, long long src_pitch,
                                 void* destination, int dw, int dh, int dcn,
                                 long long dst_pitch, const void* table,
                                 int count, long long source_batch, long long destination_batch, bool batch) {
  using namespace flagcv;
  try {
    const State& st=*static_cast<State*>(raw);
    const auto mode=static_cast<Interpolation>(st.mode);
    const GpuMat src(source,st.sw,st.sh,depth_for(st.depth),st.channels,src_pitch);
    const GpuMat dst(destination,dw,dh,depth_for(st.depth),dcn,dst_pitch);
    const DeviceWorkspace ws{table,st.host.bytes};
    LoweredOp op;
    if (st.operation==0)
      op=lower_resize(src,dst,*st.resize,ws,count,source_batch,destination_batch,batch);
    else if (st.operation==1)
      op=lower_warp_affine(src,dst,st.matrix,mode,st.border==5 ? Border::kTransparent :
                            Border::kReplicate,{},false);
    else
      op=lower_cvtcolor(src,dst,st.code,ws);
    const auto resolved=resolve_kernel("nvidia",op.kernel);

    std::ostringstream s;
    s<<std::setprecision(std::numeric_limits<float>::max_digits10);
    s<<"{\"backend\":\"nvidia\",\"target\":\"cuda:90:32\",\"kernel\":\""<<op.kernel
     <<"\",\"source\":\""<<resolved.source<<"\",\"function\":\""<<resolved.function
     <<"\",\"images\":[";
    bool first=true;
    for (const auto& image:op.images) {
      if (!first) s<<',';
      first=false;
      s<<"{\"name\":\""<<image.name<<"\",\"address\":"<<image.address
       <<",\"depth\":\""<<depth_name(image.depth)<<"\",\"channels\":"<<image.channels<<'}';
    }
    s<<"],\"params\":[";
    first=true;
    for (const auto& param:op.params) {
      if (!first) s<<',';
      first=false;
      s<<"{\"name\":\""<<param.name<<"\",\"type\":\""
       <<(param.compile_time?"constexpr":param.is_fp32?"fp32":"int")<<"\",\"value\":";
      if (param.is_fp32) s<<param.fp32_value;
      else s<<param.value;
      s<<'}';
    }
    s<<"]}";
    answer=s.str();
    return answer.c_str();
  } catch(const std::exception& e) {
    message=e.what();
    return nullptr;
  }
}

extern "C" const char* describe(void* raw, void* source, long long src_pitch,
                                void* destination, int dw, int dh, int dcn,
                                long long dst_pitch, const void* table) {
  return describe_plan(raw, source, src_pitch, destination, dw, dh, dcn, dst_pitch, table,
                       1, 0, 0, false);
}

extern "C" const char* describe_batch(void* raw, void* source, long long src_pitch,
                                      void* destination, int dw, int dh, int dcn,
                                      long long dst_pitch, const void* table, int count,
                                      long long source_batch, long long destination_batch) {
  return describe_plan(raw, source, src_pitch, destination, dw, dh, dcn, dst_pitch, table,
                       count, source_batch, destination_batch, true);
}
