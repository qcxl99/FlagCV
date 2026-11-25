// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from packages softfloat and fdlibm
// issued with the following licenses:

/*============================================================================

This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

// Adapted from OpenCV 40738fb, modules/core/src/softfloat.cpp, f64_exp.
// Gaussian only needs nonpositive arguments. Native binary64 operations
// mirror the source's individual nearest-even steps without SoftFloat types or
// an OpenCV runtime dependency. The owning source disables FP contraction.

#ifndef FLAGCV_SRC_IMGPROC_GAUSSIAN_EXP_HPP_
#define FLAGCV_SRC_IMGPROC_GAUSSIAN_EXP_HPP_

#include <array>
#include <cmath>
#include <limits>

namespace flagcv {
namespace spatial {

inline int nearest_even(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  const auto base = static_cast<int>(lower);
  return base + (fraction > 0.5 || (fraction == 0.5 && base % 2 != 0));
}

inline double gaussian_exp(double value) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  if (value < -1024.)
    return 0.;
  constexpr int table_size = 64;
  constexpr std::array<double, table_size> exp_table{
    0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
    0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0,
    0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
    0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
    0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
    0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
    0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
    0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
    0x1.6a09e667f3bcdp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
    0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.868d99b4492edp+0,
    0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0, 0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0,
    0x1.9c49182a3f090p+0, 0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
    0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
    0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0,
    0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
    0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0, 0x1.fa7c1819e90d8p+0,
  };
  constexpr double normalizer = 0x1.3ce0f3e46f431p-7;
  constexpr double inverse_ln2 = 0x1.71547652b82fep+0;
  constexpr double a0 = 0x1.5e7aa1566c2a4p-10 / normalizer;
  constexpr double a1 = 0x1.3b2a72b4f3cd3p-7 / normalizer;
  constexpr double a2 = 0x1.c6b08d81fec75p-5 / normalizer;
  constexpr double a3 = 0x1.ebfbdff82a45ap-3 / normalizer;
  constexpr double a4 = 0x1.62e42fefa39f1p-1 / normalizer;
  constexpr double a5 = 1. / normalizer;
  const double scaled = value * (inverse_ln2 * table_size);
  const int bin = nearest_even(scaled);
  int power = bin / table_size;
  int index = bin % table_size;
  if (index < 0) {
    --power;
    index += table_size;
  }
  if (power < -1022)
    return 0.;
  const double r = (scaled - bin) / table_size;
  const double polynomial = (((((a0*r + a1)*r + a2)*r + a3)*r + a4)*r + a5);
  return (std::ldexp(1., power) * normalizer * exp_table[index]) * polynomial;
}

}  // namespace spatial
}  // namespace flagcv

#endif
