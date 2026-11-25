# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Keep the Ascend source for qualification. Production builds must not
# install it while device ownership and the scratch ABI are unresolved.
message(FATAL_ERROR
  "FlagCV Ascend C++ backend is unavailable: device lifecycle and scratch ABI are unqualified. "
  "Remove ascend from FLAGCV_BACKENDS.")
