// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cuda_image.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace live {
inline bool verify_only() { return std::getenv("FLAGCV_IMGP_VERIFY_ONLY") != nullptr; }

inline void verified(const std::string& id) {
  std::cout << "VERIFY," << id << ",PASS\n";
}

class Image {
 public:
  void create(cv::Size size, int type) {
    storage_ = std::make_unique<flagcv_test::Image>(size.height, size.width, type, 1, 1, 5);
    storage_->host.setTo(cv::Scalar::all(91));
  }
  void upload(const cv::Mat& input, const flagcv_test::Stream& stream) {
    create(input.size(), input.type());
    input.copyTo(storage_->pixels());
    storage_->upload(stream.value);
  }
  void initialize(const cv::Scalar& value, const flagcv_test::Stream& stream) {
    storage_->pixels().setTo(value);
    storage_->upload(stream.value);
  }
  flagcv::GpuMat view() const { return storage_->view(); }
  std::size_t offset(cv::Rect rect) const {
    return (storage_->roi.y + rect.y) * storage_->host.step[0] +
           (storage_->roi.x + rect.x) * storage_->host.elemSize();
  }
  flagcv::GpuMat view(cv::Rect rect) const {
    const auto full = view();
    auto* ptr = static_cast<unsigned char*>(full.data());
    ptr += rect.y * full.row_pitch() + rect.x * storage_->host.elemSize();
    return {ptr, rect.width, rect.height, full.depth(), full.channels(), full.row_pitch()};
  }
  cv::Mat download(const flagcv_test::Stream& stream) const {
    const auto pixels = storage_->download(stream);
    auto guard = pixels.clone();
    storage_->host(storage_->roi).copyTo(guard(storage_->roi));
    flagcv_test::require(flagcv_test::bytes_equal(guard, storage_->host), "destination guard modified");
    return pixels(storage_->roi);
  }
 private:
  std::unique_ptr<flagcv_test::Image> storage_;
};

#ifdef FLAGCV_TEST_HIP
template <class Call>
void candidate_only(const std::string& id, Call submit, cudaStream_t stream,
                    bool host_result = false, const char* background = "repeat") {
  if (verify_only()) {
    verified(id);
    return;
  }
  using namespace std::chrono;
  using flagcv_test::cuda_check;
  using Clock = steady_clock;
  const bool long_profile = std::getenv("FLAGCV_IMGP_LONG") != nullptr;
  static constexpr int warm_group = 8;
  cudaEvent_t start{}, end{};
  struct Events {
    cudaEvent_t& start;
    cudaEvent_t& end;
    ~Events() { if (end) (void)cudaEventDestroy(end); if (start) (void)cudaEventDestroy(start); }
  } events{start, end};
  if (!host_result) {
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&end));
  }
  for (int round = 0; round < (long_profile ? 3 : 2); ++round) {
    const auto warm = Clock::now();
    do {
      for (int i = 0; i < warm_group; ++i) submit();
      cuda_check(cudaStreamSynchronize(stream));
    } while (Clock::now() - warm < milliseconds(long_profile ? 1000 : 100));
    const auto begin = Clock::now();
    struct Sample { double wall; float event; };
    std::vector<Sample> samples;
    do {
      const auto tick = Clock::now();
      submit();
      cuda_check(cudaStreamSynchronize(stream));
      const double wall = duration<double, std::micro>(Clock::now() - tick).count();
      float ms = 0;
      if (!host_result) {
        cuda_check(cudaEventRecord(start, stream));
        submit();
        cuda_check(cudaEventRecord(end, stream));
        cuda_check(cudaEventSynchronize(end));
        cuda_check(cudaEventElapsedTime(&ms, start, end));
      }
      samples.push_back({wall, ms});
    } while (Clock::now() - begin < milliseconds(100));
    const double warm_ms = duration<double, std::milli>(begin - warm).count();
    const double measure_ms = duration<double, std::milli>(Clock::now() - begin).count();
    for (const auto& sample : samples) {
      std::cout << "HIPPERF," << id << ',' << background << ',' << round << ',' << warm_ms << ','
                << measure_ms << ',' << sample.wall << ',';
      if (!host_result) std::cout << 1000 * sample.event;
      std::cout << '\n';
    }
  }
}
#endif
}  // namespace live
