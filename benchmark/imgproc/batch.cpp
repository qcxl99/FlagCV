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
  Handle handle(FLAGCV_LIVE_BACKEND, 0);
  flagcv_test::Stream stream;
  const auto native=stream.value;
#ifndef FLAGCV_TEST_HIP
  auto comparator_stream = cv::cuda::StreamAccessor::wrapStream(native);
  const auto events = live::verify_only() ? nullptr : std::make_unique<Events>();
#endif
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
      live::Image src, flag_parent;
      src.upload(inputs,stream);
      flag_parent.create({size.width+7, output_rows*row.count},type);
      flag_parent.initialize(cv::Scalar::all(91),stream);
#ifndef FLAGCV_TEST_HIP
      live::Image cuda_parent;
      std::vector<cv::cuda::GpuMat> source, cuda_out;
      for (int n=0;n<row.count;++n) {
        source.push_back(comparator_view(src.view(cv::Rect(1,1+n*source_rows,row.sw,row.sh)),type));
      }
#endif
      const ResizePlan plan({row.sw,row.sh},Depth::k8U,row.channels,
                            {{row.dw,row.dh},row.fx,row.fy,static_cast<Interpolation>(row.mode)});
      auto blob=upload(plan.table_data(),plan.table_bytes(),stream);
      const auto table=blob.view();
      const Call actual=[&] {
        resizeBatch(handle,src.view(cv::Rect(1,1,row.sw,row.sh)),
                    flag_parent.view(cv::Rect(2,2,size.width,size.height)),row.count,
                    src.view().row_pitch()*source_rows,flag_parent.view().row_pitch()*output_rows,plan,table,native);
      };
      actual();
      auto output=flag_parent.download(stream);
      for (int n=0;n<row.count;++n) {
        cv::Rect rect(2,2+n*output_rows,size.width,size.height);
        if (!flagcv_test::bytes_equal(output(rect),expected[n]))
          throw std::runtime_error("batch differs from direct CPU");
        output(rect).setTo(cv::Scalar::all(91));
      }
      if (cv::norm(output,cv::Mat(output.size(),type,cv::Scalar::all(91)),cv::NORM_INF)!=0)
        throw std::runtime_error("batch destination gap changed");
      if (!flagcv_test::bytes_equal(src.download(stream),inputs))
        throw std::runtime_error("batch source changed");
#ifdef FLAGCV_TEST_HIP
      live::candidate_only(row.id,actual,native);
#else
      if (live::verify_only()) { live::verified(row.id); continue; }
      cuda_parent.create({size.width+7,output_rows*row.count},type);
      cuda_parent.initialize(cv::Scalar::all(91),stream);
      for (int n=0; n<row.count; ++n) {
        const cv::Rect roi(2,2+n*output_rows,size.width,size.height);
        const auto view=cuda_parent.view(roi);
        require_matched_output_layout(row.id,flag_parent.view(roi),flag_parent.offset(roi),
                                      view,cuda_parent.offset(roi));
        cuda_out.push_back(comparator_view(view,type));
      }
      const Call baseline=[&] {
        for (int n=0;n<row.count;++n)
          cv::cuda::resize(source[n],cuda_out[n],{row.dw,row.dh},row.fx,row.fy,row.mode,comparator_stream);
      };
      double max_delta=0;
      bool native_valid=true;
      try {
        baseline();
        stream.wait();
        auto output=cuda_parent.download(stream);
        for (int n=0;n<row.count;++n) {
          const cv::Rect roi(2,2+n*output_rows,size.width,size.height);
          const auto native_view=cuda_parent.view(roi);
          if (cuda_out[n].data!=native_view.data() || cuda_out[n].size()!=size ||
              cuda_out[n].type()!=type || cuda_out[n].step!=static_cast<std::size_t>(native_view.row_pitch()))
            throw cv::Exception(cv::Error::StsUnmatchedFormats,"native changed batch output layout",
                                "paired-batch",__FILE__,__LINE__);
          max_delta=std::max(max_delta,cv::norm(output(roi),expected[n],cv::NORM_INF));
          output(roi).setTo(cv::Scalar::all(91));
        }
        if (cv::norm(output,cv::Mat(output.size(),type,cv::Scalar::all(91)),cv::NORM_INF)!=0)
          throw std::runtime_error("CUDA batch destination gap changed");
        if (!flagcv_test::bytes_equal(src.download(stream),inputs))
          throw std::runtime_error("batch source changed by CUDA comparator");
      } catch (const cv::Exception& e) {
        std::cout<<row.id<<",BASELINE_REFUSED,"<<std::quoted(e.err)<<'\n';
        native_valid=false;
      }
      for (int i=0;i<20;++i) {
        actual();
        if (native_valid) baseline();
      }
      stream.wait();
      for (int i=0;i<kSamples;++i) {
        if (!native_valid) {
          auto f=measure(actual,native,*events);
          std::cout<<row.id<<",CANDIDATE_ONLY,"<<i<<",F,"<<f.wall<<",,"<<f.batch<<",,"<<f.event<<",,\n";
          continue;
        }
        Sample f,c;
        if (i%2) { c=measure(baseline,native,*events); f=measure(actual,native,*events); }
        else { f=measure(actual,native,*events); c=measure(baseline,native,*events); }
        std::cout<<row.id<<(max_delta==0?",COMPARABLE,":",SEMANTICS_DIFFER,")<<i<<','<<(i%2?"CF":"FC")<<','<<f.wall<<','<<c.wall<<','
                 <<f.batch<<','<<c.batch<<','<<f.event<<','<<c.event<<','<<max_delta<<'\n';
      }
#endif
      std::cout<<std::flush;
    } catch(const std::exception& e) {
      std::cout<<row.id<<",FLAGCV_FAIL,"<<std::quoted(e.what())<<'\n';
      ++errors;
    }
  }
  return errors || !found ? 1 : 0;
}
