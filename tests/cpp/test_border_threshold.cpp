// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/flagcv.hpp>
#include <opencv2/imgproc.hpp>

#include "cuda_image.hpp"

#include <array>
#include <chrono>
#include <limits>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {
using namespace flagcv;
using namespace flagcv_test;
using TestStream = flagcv_test::Stream;

template<class Call>
void rejects(Call call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("unexpected acceptance");
}

void compare_view(const Image& image, const TestStream& stream, const cv::Mat& expected) {
  cv::Mat got=image.download(stream);
  require(bytes_equal(got(image.roi),expected), "pixels differ from OpenCV CPU");
  image.pixels().copyTo(got(image.roi));
  require(bytes_equal(got,image.host), "destination guard changed");
}

void test_borders(Handle& handle, const TestStream& native, const TestStream& default_stream) {
  constexpr BorderExtents pad{4,1,6,3};
  constexpr BorderValue value{2.5,3.5,-0.5,65503.125};
  const std::array<std::pair<Border,int>,5> modes{
      {{Border::kConstant,cv::BORDER_CONSTANT},{Border::kReplicate,cv::BORDER_REPLICATE},
       {Border::kReflect,cv::BORDER_REFLECT},{Border::kReflect101,cv::BORDER_REFLECT_101},
       {Border::kWrap,cv::BORDER_WRAP}}};
  for (int depth : {CV_8U,CV_8S,CV_16U,CV_16S,CV_32S,CV_16F,CV_32F,CV_64F}) {
    for (int cn : {1,2,3,4}) {
      const auto type=CV_MAKETYPE(depth,cn);
      Image parent(3,4,type,2,2,3), output(6,11,type,3,1,6);
      for (std::size_t index=0;index<parent.host.total()*parent.host.elemSize();++index)
        parent.host.data[index]=static_cast<unsigned char>((index*17+33)%256);
      std::memset(output.host.data,19,output.host.total()*output.host.elemSize());
      const cv::Rect roi(1,1,2,1);
      cv::Mat logical_parent=parent.pixels().clone();
      const cv::Mat logical_roi=logical_parent(roi);
      const auto p=parent.view();
      const auto bytes=depth_size(p.depth())*p.channels();
      const GpuMat src(static_cast<unsigned char*>(p.data())+roi.y*p.row_pitch()+roi.x*bytes,
                       roi.width,roi.height,p.depth(),p.channels(),p.row_pitch());
      const BorderParent parent_view{p,roi.x,roi.y};
      for (const auto& mode:modes) for (bool use_parent:{false,true}) {
        const auto& stream=use_parent?native:default_stream;
        cv::Mat reference;
        cv::copyMakeBorder(logical_roi,reference,pad.top,pad.bottom,pad.left,pad.right,
            mode.second|(use_parent?0:cv::BORDER_ISOLATED),
            cv::Scalar(value[0],value[1],value[2],value[3]));
        parent.upload(stream.value);
        output.upload(stream.value);
        copyMakeBorder(handle,src,output.view(),pad,mode.first,value,
                       use_parent?&parent_view:nullptr,stream.value);
        compare_view(output,stream,reference);
        require(bytes_equal(parent.download(stream),parent.host),"parent guard or pixels changed");
      }
    }
  }
}

void verify_signed_threshold_oracle() {
  std::array<std::int16_t,3> values{-32768,-7,9};
  const cv::Mat input(1,3,CV_16S,values.data());
  cv::Mat output;
  const auto cutoff=cv::threshold(input,output,-32769,111,cv::THRESH_TRUNC);
  require(cutoff==-32769 && cv::countNonZero(output)==0,"signed TRUNC reference changed");
}

void test_fixed_threshold(Handle& handle, const TestStream& native, const TestStream& default_stream) {
  for (int depth:{CV_8U,CV_16U,CV_16S,CV_32F,CV_64F}) for (int mode=0;mode<5;++mode) {
    Image input(2,5,CV_MAKETYPE(depth,3),1,2,7), output(2,5,CV_MAKETYPE(depth,3),3,1,2);
    cv::RNG rng(1502);
    rng.fill(input.host,cv::RNG::UNIFORM,depth==CV_8U||depth==CV_16U?0:-31,31);
    output.host.setTo(cv::Scalar::all(19));
    const TestStream& stream=mode%2?native:default_stream;
    input.upload(stream.value);
    for (double limit:{-32769.0,-32768.0,-2.5,0.0,3.0,32767.0,32768.0}) {
      cv::Mat reference;
      const double expected=cv::threshold(input.pixels(),reference,limit,11.5,mode);
      output.upload(stream.value);
      const auto actual=threshold(handle,input.view(),output.view(),limit,11.5,
                                  static_cast<ThresholdMode>(mode),false,stream.value);
      require(expected==actual,"fixed threshold return differs");
      compare_view(output,stream,reference);
    }
    require(bytes_equal(input.download(stream),input.host),"threshold changed source bytes");
  }
  Image input(1,3,CV_16S), output(1,3,CV_16S);
  const std::array<std::int16_t,3> exact{-32768,-7,9};
  std::memcpy(input.host.data,exact.data(),sizeof(exact));
  input.upload(native.value);
  std::array<double,5> limits{-32769,-32768,-32767.25,32767,32768};
  for(double cutoff:limits) {
    cv::Mat expected;
    const double ret=cv::threshold(input.pixels(),expected,cutoff,111,cv::THRESH_TRUNC);
    const auto got=threshold(handle,input.view(),output.view(),cutoff,111,ThresholdMode::kTrunc,
                             false,native.value);
    require(got==ret,"signed TRUNC return differs");
    compare_view(output,native,expected);
  }
}

void test_automatic(Handle& handle, const TestStream& native, const TestStream& default_stream) {
  for (auto method:{AutomaticThreshold::kOtsu,AutomaticThreshold::kTriangle}) {
    for (int depth:{CV_8U,CV_16U}) {
      if (method==AutomaticThreshold::kTriangle&&depth==CV_16U) continue;
      const auto dtype=depth_for(depth);
      const auto bytes=thresholdWorkspaceBytes(dtype,method);
      DeviceMemory scratch(bytes), independent(bytes);
      Image input(1,3,depth,1,2,5), output(1,3,depth,3,1,6);
      auto image=input.pixels();
      if(depth==CV_8U){image.at<std::uint8_t>(0,0)=0;image.at<std::uint8_t>(0,1)=1;image.at<std::uint8_t>(0,2)=1;}
      else {image.at<std::uint16_t>(0,0)=0;image.at<std::uint16_t>(0,1)=1024;image.at<std::uint16_t>(0,2)=65535;}
      input.upload(native.value);
      native.wait();
      const int flag=method==AutomaticThreshold::kOtsu?cv::THRESH_OTSU:cv::THRESH_TRIANGLE;
      for(int mode=0;mode<5;++mode) {
        cv::Mat expected;
        const double ret=cv::threshold(image,expected,0,42,mode|flag);
        output.host.setTo(cv::Scalar::all(19));
        output.upload(native.value);
        thresholdAuto(handle,input.view(),{},42,static_cast<ThresholdMode>(mode),
                      method,{independent.ptr,bytes},true,default_stream.value);
        const auto actual=thresholdAutoHost(handle,input.view(),output.view(),42,
                      static_cast<ThresholdMode>(mode),method,{scratch.ptr,bytes},false,native.value);
        require(ret==actual,"automatic threshold host return differs");
        compare_view(output,native,expected);
        default_stream.wait();
        std::int32_t signed_cutoff=0;
        cuda_check(cudaMemcpy(&signed_cutoff,independent.ptr,sizeof(signed_cutoff),cudaMemcpyDeviceToHost));
        require(signed_cutoff==ret,"asynchronous dry-run result differs");
      }
      rejects([&] { thresholdAuto(handle,input.view(),output.view(),42,ThresholdMode::kBinary,
                         method,{scratch.ptr,bytes-1},false,native.value); });
      rejects([&] { thresholdAuto(handle,input.view(),output.view(),42,ThresholdMode::kBinary,
                         method,{input.view().data(),bytes},false,native.value); });
    }
  }
}
void test_float_tails(Handle& handle, const TestStream& stream) {
  for(int depth:{CV_32F,CV_64F}) for(bool padded:{false,true}) {
    Image input(2,5,depth,padded?1:0,padded?2:0,padded?3:0);
    Image output(2,5,depth,padded?2:0,padded?1:0,padded?5:0);
    auto image=input.pixels();
    const std::array<double,5> values{std::numeric_limits<double>::quiet_NaN(),-0.,0.,1.,
                                      std::numeric_limits<double>::quiet_NaN()};
    for(int y=0;y<2;++y) for(int x=0;x<5;++x) {
      if(depth==CV_32F) image.at<float>(y,x)=static_cast<float>(values[x]);
      else image.at<double>(y,x)=values[x];
    }
    output.host.setTo(cv::Scalar(19));
    input.upload(stream.value);
    output.upload(stream.value);
    cv::Mat reference_backing=output.host.clone();
    cv::Mat reference=reference_backing(output.roi);
    cv::threshold(image,reference,-0.,1.,cv::THRESH_TRUNC);
    threshold(handle,input.view(),output.view(),-0.,1.,ThresholdMode::kTrunc,false,stream.value);
    compare_view(output,stream,reference);
  }
}

template<class Call>
void time_calls(const char* label,const TestStream& stream,Call call) {
  constexpr int samples=5, repeats=20;
  call();
  stream.wait();
#ifdef FLAGCV_TEST_HIP
  using Event=hipEvent_t;
  auto create=hipEventCreate;
  auto destroy=hipEventDestroy;
  auto record=hipEventRecord;
  auto elapsed=hipEventElapsedTime;
#else
  using Event=cudaEvent_t;
  auto create=cudaEventCreate;
  auto destroy=cudaEventDestroy;
  auto record=cudaEventRecord;
  auto elapsed=cudaEventElapsedTime;
#endif
  Event begin{},end{};
  cuda_check(create(&begin));
  cuda_check(create(&end));
  for(int sample=0;sample<samples;++sample) {
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;++i) {call(); stream.wait();}
    const auto stop=std::chrono::steady_clock::now();
    const double wall=std::chrono::duration<double,std::micro>(stop-start).count()/repeats;
    cuda_check(record(begin,stream.value));
    for(int i=0;i<repeats;++i) call();
    cuda_check(record(end,stream.value));
    stream.wait();
    float event=0;
    cuda_check(elapsed(&event,begin,end));
    std::cout<<label<<" sample="<<sample<<" synchronized_wall_us="<<wall
             <<" repeated_device_event_us="<<event*1000/repeats<<'\n';
  }
  cuda_check(destroy(end));
  cuda_check(destroy(begin));
}

void measure(Handle& handle,const TestStream& stream) {
  Image input(1080,1920,CV_8UC1), output(1080,1920,CV_8UC1), padded(1086,1926,CV_8UC1);
  cv::RNG rng(820);
  rng.fill(input.host,cv::RNG::UNIFORM,0,256);
  output.host.setTo(cv::Scalar(19));
  padded.host.setTo(cv::Scalar(19));
  input.upload(stream.value);
  const auto border=[&]{copyMakeBorder(handle,input.view(),padded.view(),{3,3,3,3},
                                       Border::kReflect,{},nullptr,stream.value);};
  const auto fixed=[&]{threshold(handle,input.view(),output.view(),126,255,
                                 ThresholdMode::kBinary,false,stream.value);};
  cv::Mat reference;
  border();
  cv::copyMakeBorder(input.pixels(),reference,3,3,3,3,cv::BORDER_REFLECT);
  compare_view(padded,stream,reference);
  fixed();
  cv::threshold(input.pixels(),reference,126,255,cv::THRESH_BINARY);
  compare_view(output,stream,reference);
  time_calls("copyMakeBorder U8 C1 1080x1920 +3 reflect",stream,border);
  time_calls("threshold U8 C1 1080x1920 binary",stream,fixed);
}

}  // namespace

int main(int argc,char** argv) {
  try {
    cv::setNumThreads(1);
    verify_signed_threshold_oracle();
    if(argc==2&&std::string_view(argv[1])=="--oracle") return 0;
#ifdef FLAGCV_TEST_HIP
    Handle handle("hip");
#else
    Handle handle;
#endif
    const TestStream native, default_stream(false);
    test_borders(handle,native,default_stream);
    test_fixed_threshold(handle,native,default_stream);
    test_float_tails(handle,native);
    test_automatic(handle,native,default_stream);
    if(argc==2&&std::string_view(argv[1])=="--timing") measure(handle,native);
    std::cout<<"copyMakeBorder and threshold public CPU comparisons passed\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
