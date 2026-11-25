// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Adapted from modules/imgproc/src/resize.cpp; retain its original license notice.
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// Parameter resolution and interpolation tables follow the generic cv::resize
// implementation in modules/imgproc/src/resize.cpp. Float32 multiply/add operations
// must round separately (-ffp-contract=off).
// Standalone CPU controls and test support do not use library-private headers.
#include "resize_tables.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "resize tables require IEEE single-precision float evaluation (FLT_EVAL_METHOD == 0)"
#endif
namespace flagcv::resize_tables
{

namespace
{

constexpr int COEF_BITS = 11;
constexpr int COEF_SCALE = 1 << COEF_BITS;
inline float f32(double v)
{
    return static_cast<float>(v);
}

// rint with ties to even (the default rounding mode), as cvRound.
inline double rint_even(double v)
{
    const double lower = std::floor(v);
    const double fraction = v - lower;
    return lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0));
}

inline int sat_short(float v)
{ // saturate_cast<short>(float)
    double r = rint_even(v);
    return static_cast<int>(std::max(-32768.0, std::min(32767.0, r)));
}

void cubic_coeffs(float x, float *c)
{ // interpolateCubic, float32 op by op
    const float A = -0.75f, one = 1.0f;
    float xp1 = x + one;
    float c0 = ((A * xp1 - 5.0f * A) * xp1 + 8.0f * A) * xp1 - 4.0f * A;
    float c1 = ((A + 2.0f) * x - (A + 3.0f)) * x * x + one;
    float omx = one - x;
    float c2 = ((A + 2.0f) * omx - (A + 3.0f)) * omx * omx + one;
    float c3 = one - c0 - c1 - c2;
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
    c[3] = c3;
}

} // namespace

// ------------------------------------------------------------------ parameters

Size2 resolve_size(int src_h, int src_w, int dst_w, int dst_h, double fx, double fy)
{ // resolve_size
    if (dst_w > 0 && dst_h > 0)
        return {dst_h, dst_w, static_cast<double>(dst_h) / src_h,
                static_cast<double>(dst_w) / src_w};
    if (!(fx > 0 && fy > 0))
        throw std::invalid_argument("either dsize or both fx and fy (> 0) must be given");
    return {static_cast<int>(rint_even(src_h * fy)), static_cast<int>(rint_even(src_w * fx)), fy,
            fx};
}

RouteInfo route(int src_h, int src_w, int dst_h, int dst_w, double inv_y, double inv_x, int interp)
{ // route
    if (interp < INTER_NEAREST || interp > INTER_AREA)
        throw std::invalid_argument("unsupported interpolation");
    RouteInfo r;
    if (dst_h == src_h && dst_w == src_w)
    {
        r.kind = ROUTE_COPY;
        return r;
    }
    if (interp == INTER_NEAREST)
    {
        r.kind = ROUTE_NEAREST;
        return r;
    }
    double scale_x = 1.0 / inv_x, scale_y = 1.0 / inv_y;
    const double rounded_x = rint_even(scale_x), rounded_y = rint_even(scale_y);
    const bool area_fast = rounded_x >= 1 && rounded_y >= 1 &&
        rounded_x <= std::numeric_limits<int>::max() &&
        rounded_y <= std::numeric_limits<int>::max() &&
        std::fabs(scale_x - rounded_x) < DBL_EPSILON &&
        std::fabs(scale_y - rounded_y) < DBL_EPSILON;
    int mode = interp;
    if (mode == INTER_LINEAR && area_fast && rounded_x == 2 && rounded_y == 2)
        mode = INTER_AREA;
    if (mode == INTER_AREA && scale_x >= 1 && scale_y >= 1)
    {
        if (area_fast)
        {
            r.kind = ROUTE_AREA_FAST;
            r.ix = static_cast<int>(rounded_x);
            r.iy = static_cast<int>(rounded_y);
            return r;
        }
        r.kind = ROUTE_AREA;
        return r;
    }
    r.kind = mode == INTER_CUBIC ? ROUTE_CUBIC : ROUTE_LINEAR;
    r.area_mode = mode == INTER_AREA;
    return r;
}

// ---------------------------------------------------------------- coefficients

Axis generic_axis(int dst, int src, double inv_scale, int ksize, bool area_mode, bool horizontal)
{ // generic_axis
    Axis a;
    a.ksize = ksize;
    a.xmin = 0;
    a.xmax = dst;
    if (dst > std::numeric_limits<int>::max() / ksize)
        throw std::invalid_argument("resize coefficients exceed int32 element indexing");
    a.ofs.resize(dst);
    a.coeffs.resize(static_cast<std::size_t>(dst) * ksize);
    const double scale = 1.0 / inv_scale;
    const int ksize2 = ksize / 2;
    const bool cubic = ksize == 4;
    for (int d = 0; d < dst; ++d)
    {
        float f;
        std::int64_t s;
        if (!area_mode)
        {
            f = f32((d + 0.5) * scale - 0.5);
            s = static_cast<std::int64_t>(std::floor(f));
            f = f - static_cast<float>(s);
        }
        else
        {
            s = static_cast<std::int64_t>(std::floor(d * scale));
            f = f32((d + 1) - (s + 1) * inv_scale);
            f = f <= 0 ? 0.0f : f - static_cast<float>(std::floor(f));
        }
        if (horizontal)
        {
            if (s < ksize2 - 1)
            {
                a.xmin = d + 1;
                if (s < 0 && !cubic)
                {
                    f = 0.0f;
                    s = 0;
                }
            }
            if (s + ksize2 >= src)
            {
                a.xmax = std::min(a.xmax, d);
                if (s >= src - 1 && !cubic)
                {
                    f = 0.0f;
                    s = src - 1;
                }
            }
        }
        // If float32 rounded the last base past INT_MAX, its taps still
        // clamp to the same last source pixel.
        a.ofs[d] = static_cast<int>(std::min<std::int64_t>(s, std::numeric_limits<int>::max()));
        float *c = &a.coeffs[static_cast<std::size_t>(d) * ksize];
        if (cubic)
            cubic_coeffs(f, c);
        else
        {
            c[0] = 1.0f - f;
            c[1] = f;
        }
    }
    return a;
}

std::vector<int> fixed_coeffs(const std::vector<float> &coeffs)
{ // fixed
    std::vector<int> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i)
        out[i] = sat_short(coeffs[i] * static_cast<float>(COEF_SCALE));
    return out;
}

namespace {

struct AreaCell {
    int first;
    int last;
    double begin;
    double end;
    double width;

    bool partial_first() const { return first - begin > 1e-3; }
    bool partial_last() const { return end - last > 1e-3; }
    int count() const { return last - first + partial_first() + partial_last(); }
};

AreaCell area_cell(int src, int d, double scale)
{
    const double begin = d * scale, end = begin + scale;
    const auto last = static_cast<int>(std::min(std::floor(end), double(src - 1)));
    const auto first = static_cast<int>(std::min(std::ceil(begin), double(last)));
    return {first, last, begin, end, std::min(scale, src - begin)};
}

} // namespace

AreaTab area_tab(int src, int dst, double scale)
{
    AreaTab out;
    for (int d = 0; d < dst; ++d)
        out.taps = std::max(out.taps, area_cell(src, d, scale).count());
    if (out.taps <= 0 || dst > std::numeric_limits<int>::max() / out.taps)
        throw std::invalid_argument("resize AREA tables exceed int32 element indexing");
    const auto slots = static_cast<std::size_t>(dst) * out.taps;
    out.src_index.resize(slots);
    out.alpha.resize(slots);
    out.count.resize(dst);
    for (int d = 0; d < dst; ++d)
    {
        const AreaCell cell = area_cell(src, d, scale);
        const auto begin = static_cast<std::size_t>(d) * out.taps;
        auto offset = begin;
        const auto append = [&](int index, double alpha) {
            out.src_index[offset] = index;
            out.alpha[offset++] = f32(alpha);
        };
        if (cell.partial_first())
            append(cell.first - 1, (cell.first - cell.begin) / cell.width);
        for (int x = cell.first; x < cell.last; ++x)
            append(x, 1.0 / cell.width);
        if (cell.partial_last())
            append(cell.last, std::min(std::min(cell.end - cell.last, 1.0), cell.width) / cell.width);
        out.count[d] = static_cast<int>(offset - begin);
    }
    return out;
}

} // namespace flagcv::resize_tables
