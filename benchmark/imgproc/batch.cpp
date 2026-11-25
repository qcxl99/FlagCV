#define main single_image_driver
#include "first.cpp"
#undef main

struct BatchCase { const char* id; int count,channels,mode,sw,sh,dw,dh; double fx,fy; };
const BatchCase batch_cases[] = {
#include "batch_cases.inc"
};

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  cv::setNumThreads(1);
  Handle handle;
  cv::cuda::Stream stream(cudaStreamNonBlocking);
  const auto native=cv::cuda::StreamAccessor::getStream(stream);
  Events events;
  std::cout<<"id,status,sample,order,flagcv_wall_us,cuda_wall_us,flagcv_batch_wall_us,"
             "cuda_batch_wall_us,flagcv_event_us,cuda_event_us,cuda_cpu_max_abs\n";
  int errors=0;
  bool found=false;
  for (const auto& row:batch_cases) {
    if (std::string_view(argv[1]) != row.id) continue;
    found=true;
    try {
      const int type=CV_MAKETYPE(CV_8U,row.channels);
      const int source_rows=row.sh+3;
      cv::Mat inputs(source_rows*row.count,row.sw+5,type);
      cv::RNG rng(809);
      rng.fill(inputs,cv::RNG::UNIFORM,0,256);
      std::vector<cv::Mat> expected;
      for (int n=0;n<row.count;++n) {
        cv::Mat out;
        cv::resize(inputs(cv::Rect(1,1+n*source_rows,row.sw,row.sh)),out,
                   {row.dw,row.dh},row.fx,row.fy,row.mode);
        expected.push_back(out);
      }
      const auto size=expected.front().size();
      const int output_rows=size.height+5;
      cv::cuda::GpuMat src,flag_parent(output_rows*row.count,size.width+7,type),
                       cuda_parent(output_rows*row.count,size.width+7,type);
      src.upload(inputs,stream);
      std::vector<cv::cuda::GpuMat> source,flag_out,cuda_out;
      for (int n=0;n<row.count;++n) {
        source.push_back(src(cv::Rect(1,1+n*source_rows,row.sw,row.sh)));
        cv::Rect rect(2,2+n*output_rows,size.width,size.height);
        flag_out.push_back(flag_parent(rect));
        cuda_out.push_back(cuda_parent(rect));
      }
      const ResizePlan plan({row.sw,row.sh},Depth::k8U,row.channels,
                            {{row.dw,row.dh},row.fx,row.fy,static_cast<Interpolation>(row.mode)});
      auto blob=upload(plan.table_data(),plan.table_bytes(),stream);
      const auto table=workspace(blob);
      const Call actual=[&] {
        resizeBatch(handle,view(source[0]),view(flag_out[0]),row.count,src.step*source_rows,
                    flag_parent.step*output_rows,plan,table,native);
      };
      const Call baseline=[&] {
        for (int n=0;n<row.count;++n)
          cv::cuda::resize(source[n],cuda_out[n],{row.dw,row.dh},row.fx,row.fy,row.mode,stream);
      };
      actual();
      stream.waitForCompletion();
      for (int n=0;n<row.count;++n) {
        cv::Mat got;
        flag_out[n].download(got);
        if (cv::norm(got,expected[n],cv::NORM_INF)!=0)
          throw std::runtime_error("batch differs from direct CPU");
      }
      double max_delta=0;
      bool native_valid=true;
      try {
        baseline();
        stream.waitForCompletion();
        for (int n=0;n<row.count;++n) {
          cv::Mat got;
          cuda_out[n].download(got);
          max_delta=std::max(max_delta,cv::norm(got,expected[n],cv::NORM_INF));
        }
      } catch (const cv::Exception& e) {
        std::cout<<row.id<<",BASELINE_REFUSED,"<<std::quoted(e.err)<<'\n';
        native_valid=false;
      }
      for (int i=0;i<20;++i) {
        actual();
        if (native_valid) baseline();
      }
      stream.waitForCompletion();
      for (int i=0;i<kSamples;++i) {
        if (!native_valid) {
          auto f=measure(actual,native,events);
          std::cout<<row.id<<",CANDIDATE_ONLY,"<<i<<",F,"<<f.wall<<",,"<<f.batch<<",,"<<f.event<<",,\n";
          continue;
        }
        Sample f,c;
        if (i%2) { c=measure(baseline,native,events); f=measure(actual,native,events); }
        else { f=measure(actual,native,events); c=measure(baseline,native,events); }
        std::cout<<row.id<<(max_delta==0?",COMPARABLE,":",SEMANTICS_DIFFER,")<<i<<','<<(i%2?"CF":"FC")<<','<<f.wall<<','<<c.wall<<','
                 <<f.batch<<','<<c.batch<<','<<f.event<<','<<c.event<<','<<max_delta<<'\n';
      }
      std::cout<<std::flush;
    } catch(const std::exception& e) {
      std::cout<<row.id<<",FLAGCV_FAIL,"<<std::quoted(e.what())<<'\n';
      ++errors;
    }
  }
  return errors || !found ? 1 : 0;
}
