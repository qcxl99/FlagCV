# FlagCV

FlagCV provides immediate-mode GPU image processing through OpenCV-shaped C++
interfaces and a standalone runtime. The NVIDIA delivery covers add, subtract,
multiply, bitwise AND, bitwise NOT, and flip. The public library uses caller-owned
image memory and native streams. Triton compilation runs in a separate Python
process.

FlagCV is part of FlagOS. Kernel support, backend validation and the development
queue are tracked in [the operator inventory](docs/operators.md). For a change
to this standalone repository, use [the FlagCV merge requirements](CONTRIBUTING.md).

## Quick Start

### Requirements

On Linux, the runtime build uses:

- CMake 3.23+ and a C++17 compiler
- Python 3.10+ for the merge checker, with upstream Triton for NVIDIA or the backend's supported [FlagTree/Triton distribution](https://github.com/flagos-ai/flagtree).

Plus the toolchain of the backend you build. `libflagcv` itself links no vendor SDK and no CUDA: it reaches the device only through a `dlopen`'d plugin, so selecting a backend never changes the core.

| `FLAGCV_BACKENDS` entry | Plugin links |
|---|---|
| `nvidia` | CUDA driver API (`CUDA::cuda_driver`) |
| `iluvatar` | CUDA-compatible driver API (`CUDA::cuda_driver`) |
| `hip` | HIP runtime from ROCm or DTK |
| `hygon` | Legacy name for the shared HIP plugin |
| `mthreads` | MUSA runtime |
| `thead` | PPU driver API (`CUstream`), T-Head PPU SDK |

NVIDIA and the shared HIP path have on-device public API evidence. HIP
qualification is tracked per device and toolchain. Iluvatar, MThreads and
T-Head can be built with their SDKs and await native launch qualification.
For Iluvatar, use `-DCUDAToolkit_ROOT=<corex>`.

C++ interface conventions and the call path are documented in
[C++ operator interfaces](docs/cpp-operators.md).

Ascend C++ is unavailable in this build. Selecting `ascend` in
`FLAGCV_BACKENDS` fails at configure time. Enablement requires verification
with an application-owned AscendCL context, buffers and stream.

The shared HIP backend uses the same module and stream API on AMD ROCm and
Hygon DTK. See [HIP validation](docs/hip-validation.md) for the device matrix
and [scratch ABI qualification](docs/architecture.md#scratch-abi-qualification)
for the native launch boundary.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLAGCV_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build -L host --output-on-failure
```

For a host-only check without a vendor SDK, use
`python tools/check_merge.py --profile host --output /path/to/new-host-evidence`.
The installed consumer uses the shared Runtime for automatic discovery of
relocated plugins and data. Static linking can use `FLAGCV_DATA_DIR` and
`FLAGCV_PLUGIN_DIR` to select those installed resources explicitly.

Enable `FLAGCV_TEST_NVIDIA` to build the public C++ API and backend failure tests with OpenCV CPU 5. See [NVIDIA validation](docs/nvidia-validation.md) for the complete test and benchmark commands.

### Use from C++

Install, then point your project at the install prefix:

```sh
cmake --install build --prefix /opt/flagcv
```

The default shared library locates its installed backend plugins and kernel sources relative to its own location.

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.23)
project(flip_demo LANGUAGES CXX)

find_package(FlagCV REQUIRED)
find_package(CUDAToolkit REQUIRED)  # only for cudaMalloc in this demo

add_executable(flip_demo main.cpp)
target_link_libraries(flip_demo PRIVATE FlagCV::flagcv CUDA::cudart)
```

`main.cpp`:

```cpp
#include <flagcv/flagcv.hpp>

#include <cuda_runtime.h>

#include <cstdio>

int main() {
  const int width = 4, height = 2;
  unsigned char host[] = {1, 2, 3, 4,
                          5, 6, 7, 8};

  // FlagCV never allocates device memory: the caller owns both buffers.
  void *src = nullptr, *dst = nullptr;
  cudaMalloc(&src, sizeof(host));
  cudaMalloc(&dst, sizeof(host));
  cudaMemcpy(src, host, sizeof(host), cudaMemcpyHostToDevice);

  flagcv::Handle handle("nvidia", 0);  // backend plugin + kernel cache
  flagcv::GpuMat in(src, width, height, flagcv::Depth::k8U, 1);
  flagcv::GpuMat out(dst, width, height, flagcv::Depth::k8U, 1);
  flagcv::flip(handle, in, out, 1);  // 1: mirror columns, 0: rows, -1: both

  // flip is enqueued on the default stream; this copy waits for it.
  cudaMemcpy(host, dst, sizeof(host), cudaMemcpyDeviceToHost);
  for (unsigned char v : host) std::printf("%d ", v);  // 4 3 2 1 8 7 6 5
  std::printf("\n");

  cudaFree(src);
  cudaFree(dst);
}
```

```sh
cmake -B build -DCMAKE_PREFIX_PATH=/opt/flagcv
cmake --build build
./build/flip_demo
```

The first call for a kernel variant starts the compiler if needed and loads
the resulting module. Cached calls enqueue work on the selected stream.
See [the architecture](docs/architecture.md) for cache keys and ownership.

### Correctness and performance

OpenCV CPU defines operator semantics. The C++ API and Python kernel have
separate NVIDIA benchmarks against same-device OpenCV CUDA. Build the bridge
and use the commands and acceptance reports in
[NVIDIA validation](docs/nvidia-validation.md). Python candidate injection and
FlagGems-compatible reporting are documented in [Python evaluation](docs/python-evaluation.md).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FLAGCV_PYTHON` | `python3` | Interpreter for the external compiler |
| `FLAGCV_COMPILER_TIMEOUT_MS` | `120000` | Deadline for a compiler request, in milliseconds |
| `FLAGCV_CACHE_DIR` | `~/.cache/flagcv` | Where compiled kernel binaries are written |
| `FLAGCV_DATA_DIR` | `<prefix>/share/flagcv`, else the source tree | Kernel sources, compiler and platform providers |
| `FLAGCV_PLUGIN_DIR` | `<prefix>/lib/flagcv`, else the build tree | Backend plugins (`libflagcv_<backend>.so`) |

See [docs/architecture.md](docs/architecture.md) for how the runtime, the compiler process and the plugins fit together, and [docs/operator-development.md](docs/operator-development.md) for adding an operator.
