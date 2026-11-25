/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_IMGPROC_HPP_
#define FLAGCV_IMGPROC_HPP_

#include <flagcv/gpumat.hpp>
#include <flagcv/handle.hpp>

#include <array>
#include <memory>
#include <optional>

/*
 * Colour conversion, geometric transforms, filtering, morphology,
 * thresholding, histograms. Mirrors opencv2/imgproc.hpp.
 */
namespace flagcv {

// A row-major 2x3 transform.
using AffineMatrix = std::array<double, 6>;

/*
 * Sample src into the preallocated dst view. Use dst dimensions to set the
 * output size. With inverse_map, matrix maps dst coordinates directly to src.
 * Source and destination must have the same pixel type and disjoint storage.
 *
 * Supports 8U, 16U and 32F with 1, 3 or 4 packed channels and independent
 * row pitches. Area is an alias for linear. With a transparent border,
 * initialized destination pixels supply missing linear taps and wholly
 * outside pixels retain their previous value. Cubic disallows transparency.
 * Keep the handle and views alive until the supplied stream completes.
 */
void warpAffine(Handle& handle, const GpuMat& src, const GpuMat& dst,
                const AffineMatrix& matrix,
                Interpolation interpolation = Interpolation::kLinear,
                Border border = Border::kConstant,
                const BorderValue& border_value = {},
                bool inverse_map = false, Stream stream = nullptr);

// Positive width and height select an explicit size. {0, 0} selects scales.
struct Size {
  int width = 0;
  int height = 0;
};

// Immutable bytes already uploaded by the caller to this Handle's device.
struct DeviceWorkspace {
  const void* data = nullptr;
  std::size_t bytes = 0;
};

struct ResizeOptions {
  Size size;
  double fx = 0;
  double fy = 0;
  Interpolation interpolation = Interpolation::kLinear;
};

/*
 * Builds the host interpolation tables once. The caller copies table_data()
 * to device storage and orders the upload before any stream that uses it.
 * Keep the host data alive through upload and the device data alive and
 * immutable until every use completes. Some routes have zero table bytes.
 */
class ResizePlan {
 public:
  ResizePlan(Size source, Depth depth, int channels, ResizeOptions options);
  ResizePlan(const ResizePlan&) = default;
  ResizePlan& operator=(const ResizePlan&) = default;

  Size destination_size() const;
  const void* table_data() const;
  std::size_t table_bytes() const;

 private:
  struct Data;
  std::shared_ptr<const Data> data_;
  friend struct ResizeAccess;
};

// The images and any workspace are preallocated. Calls enqueue on stream.
void resize(Handle& handle, const GpuMat& src, const GpuMat& dst,
            const ResizePlan& plan, DeviceWorkspace workspace = {},
            Stream stream = nullptr);

// A batch contains equal-sized 8U images. Batch pitches are independent bytes.
void resizeBatch(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 int count, std::int64_t src_batch_pitch, std::int64_t dst_batch_pitch,
                 const ResizePlan& plan, DeviceWorkspace workspace = {},
                 Stream stream = nullptr);

// Table families are shareable by every code that names the same family.
enum class ColorTables { kNone, kHsv, kHsvFull, kLabLuvReverse, kLuvForward };

struct HostTableData {
  const void* data;
  std::size_t bytes;
};

struct ColorOutput {
  Size size;
  int channels;
};

// OpenCV numeric codes with their original output channel count and depth.
ColorOutput cvtColorOutput(Size source, Depth depth, int channels, int code);
ColorTables cvtColorTableKind(int code);
HostTableData cvtColorTableData(ColorTables kind);

/*
 * Upload the exact payload for cvtColorTableKind(code) before first use.
 * The caller owns the device copy, its stream visibility and its lifetime.
 * A table-free conversion takes an empty workspace.
 */
void cvtColor(Handle& handle, const GpuMat& src, const GpuMat& dst, int code,
              DeviceWorkspace workspace = {}, Stream stream = nullptr);

enum class ThresholdMode { kBinary, kBinaryInv, kTrunc, kToZero, kToZeroInv };
enum class AutomaticThreshold { kOtsu, kTriangle };

// Fixed modes return the CPU-equivalent threshold without synchronizing the
// device. Dry-run evaluates the return value without writing dst.
double threshold(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 double cutoff, double maxval, ThresholdMode mode,
                 bool dry_run = false, Stream stream = nullptr);

struct ThresholdScratch {
  void* data = nullptr;
  std::size_t bytes = 0;
};

std::size_t thresholdWorkspaceBytes(Depth depth, AutomaticThreshold method);

// The first int32 in scratch receives the signed automatic threshold. Keep
// the exclusive, disjoint scratch, images, handle and stream alive until
// queued work has completed, including if a later step throws.
void thresholdAuto(Handle& handle, const GpuMat& src, const GpuMat& dst, double maxval,
                   ThresholdMode mode, AutomaticThreshold method, ThresholdScratch scratch,
                   bool dry_run = false, Stream stream = nullptr);

// Same operation with a synchronized host result. Waits for the chosen
// stream, then performs the backend's blocking device-to-host readback.
double thresholdAutoHost(Handle& handle, const GpuMat& src, const GpuMat& dst, double maxval,
                         ThresholdMode mode, AutomaticThreshold method, ThresholdScratch scratch,
                         bool dry_run = false, Stream stream = nullptr);

struct GaussianOptions {
  Size kernel;
  double sigma_x = 0;
  double sigma_y = 0;
  Border border = Border::kReflect101;
};

// The prepared Gaussian is fixed to source size and depth. Axes are odd and
// at most 65535. A nonpositive axis with positive sigma selects the
// OpenCV-derived size. A nonpositive sigma_y inherits sigma_x. Integer U8/U16
// use the fixed CPU quantization and rounding.
class GaussianPlan {
 public:
  GaussianPlan(Size source, Depth depth, GaussianOptions options);
  Size kernel_size() const;
  Depth scratch_depth() const;
  HostTableData table() const;

 private:
  struct Data;
  std::shared_ptr<const Data> data_;
  friend struct SpatialFilterAccess;
};

struct FilterAnchor {
  int x = -1;
  int y = -1;
};

struct Filter2DOptions {
  std::optional<Depth> ddepth;
  FilterAnchor anchor{};
  double delta = 0;
  Border border = Border::kReflect101;
};

// Copies a row-major coefficient matrix. The operation is direct
// correlation, including for large kernels. An absent ddepth preserves the
// source depth, and -1 in either axis of the anchor selects that axis's center.
class Filter2DPlan {
 public:
  Filter2DPlan(Size kernel, const double* coefficients, Depth source,
               Filter2DOptions options = {});
  HostTableData table() const;

 private:
  struct Data;
  std::shared_ptr<const Data> data_;
  friend struct SpatialFilterAccess;
};

// Both filters use the ROI as an isolated image, support 1–4 packed channels
// and require disjoint input/output pixels. The caller uploads the plan table
// to aligned immutable device workspace and orders it before submitting.
void GaussianBlur(Handle& handle, const GpuMat& src, const GpuMat& dst,
                  const GaussianPlan& plan, DeviceWorkspace workspace,
                  Stream stream = nullptr);
// The two-pass overload uses a writable scratch image of the source size and
// channel count, with scratch_depth(). Keep it disjoint from both images and
// the coefficient workspace. The two launches are ordered on stream.
void GaussianBlur(Handle& handle, const GpuMat& src, const GpuMat& dst,
                  const GaussianPlan& plan, DeviceWorkspace workspace,
                  const GpuMat& scratch, Stream stream = nullptr);
void filter2D(Handle& handle, const GpuMat& src, const GpuMat& dst,
              const Filter2DPlan& plan, DeviceWorkspace workspace,
              Stream stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_IMGPROC_HPP_
