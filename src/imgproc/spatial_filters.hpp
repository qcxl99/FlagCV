/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGCV_SRC_IMGPROC_SPATIAL_FILTERS_HPP_
#define FLAGCV_SRC_IMGPROC_SPATIAL_FILTERS_HPP_

#include <flagcv/imgproc.hpp>

#include <vector>

namespace flagcv {

struct GaussianPlan::Data {
  Size source;
  Size kernel;
  Depth depth;
  Border border;
  std::vector<double> coefficients;
};

struct Filter2DPlan::Data {
  Size kernel;
  Depth source_depth;
  Depth destination_depth;
  Border border;
  FilterAnchor anchor;
  bool zero_filter;
  bool all_normal_f32_taps;
  bool fp32_integer_exact;
  bool fp64_u16_exact;
  bool fp64_u16_pair_exact;
  bool has_nonzero_delta;
  std::vector<double> coefficients;
};

struct SpatialFilterAccess {
  static const GaussianPlan::Data& get(const GaussianPlan& plan) { return *plan.data_; }
  static const Filter2DPlan::Data& get(const Filter2DPlan& plan) { return *plan.data_; }
};

}  // namespace flagcv

#endif  // FLAGCV_SRC_IMGPROC_SPATIAL_FILTERS_HPP_
