# FlagCV

## About

FlagCV is part of [FlagOS](https://flagos.io/), a fully open-source system software stack designed to unify the model–system–chip layers and foster an open and collaborative ecosystem.

FlagCV is a high-performance, cross-platform computer vision operator library implemented in the [Triton](https://github.com/triton-lang/triton) language. It aims to provide optimized implementations of common image processing operators, including color conversion, filtering, geometric transformations, and arithmetic operations, through OpenCV-style C++ APIs.

Built on [FlagTree](https://github.com/flagos-ai/flagtree), FlagCV uses backend-neutral kernels and hardware-specific optimizations to accelerate computer vision workloads across diverse hardware platforms.

## Quick Start

### Requirements

- Linux
- CMake 3.23+ and a C++17 compiler
- Python 3.10+ with the corresponding FlagTree distribution
- SDK and toolchain for the selected hardware backend

FlagCV compiles Triton kernels at runtime and accesses hardware through pluggable backends. The core library is independent of vendor-specific SDKs.

### Hardware Backends

| `FLAGCV_BACKENDS` entry | Plugin links |
|---|---|
| `nvidia` | CUDA driver API |
| `amd` | ROCm HIP runtime |
| `hygon` | DTK HIP runtime |
| `iluvatar` | CUDA-compatible driver API (CoreX) |
| `mthreads` | MThreads MUSA runtime |
| `thead` | T-Head PPU driver API |
| `ascend` | Unavailable. AscendCL integration is planned. |

See the [operator inventory](docs/operators.md) for current operators, validated devices, and preview backends.

## Build and Test

Select the backend through `FLAGCV_BACKENDS`:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=nvidia \
  -DFLAGCV_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install FlagCV:

```sh
cmake --install build --prefix /opt/flagcv
```

This example runs the default test suite. See [Testing](docs/testing.md) for the host-only workflow, device tests, and benchmark instructions.

### Use from C++

FlagCV provides OpenCV-style C++ interfaces with explicit device selection and optional native streams.

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.23)
project(flip_demo LANGUAGES CXX)

find_package(FlagCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

  cudaSetDevice(0);  // Select the CUDA device for the caller's allocations.
  flagcv::setDevice("nvidia", 0);  // Select FlagCV's logical device.

  void *src = nullptr, *dst = nullptr;
  cudaMalloc(&src, sizeof(host));
  cudaMalloc(&dst, sizeof(host));
  cudaMemcpy(src, host, sizeof(host), cudaMemcpyHostToDevice);

  flagcv::GpuMat in(src, width, height, flagcv::Depth::k8U, 1);
  flagcv::GpuMat out(dst, width, height, flagcv::Depth::k8U, 1);

  flagcv::flip(in, out, 1);

  cudaMemcpy(host, dst, sizeof(host), cudaMemcpyDeviceToHost);
  for (unsigned char v : host)
    std::printf("%d ", v);
  std::printf("\n");

  cudaFree(src);
  cudaFree(dst);
}
```

Build and run:

```sh
cmake -B build -DCMAKE_PREFIX_PATH=/opt/flagcv
cmake --build build
./build/flip_demo
```

Expected output:

```text
4 3 2 1 8 7 6 5
```

For explicit device and stream management, use a `Handle`:

```cpp
flagcv::Handle handle("nvidia", 0);
flagcv::flip(handle, in, out, 1, stream);
```

Both interfaces share the same runtime and kernel implementation. Compiled kernels are cached for subsequent execution.

See [C++ Operator Interfaces](docs/cpp-operators.md) for more information.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `FLAGCV_PYTHON` | `python3` | Python interpreter for kernel compilation |
| `FLAGCV_CACHE_DIR` | `~/.cache/flagcv` | Compiled kernel cache |
| `FLAGCV_DATA_DIR` | Install prefix or source tree | Kernel and compiler resources |
| `FLAGCV_PLUGIN_DIR` | Install prefix or build tree | Backend plugins |
| `FLAGCV_COMPILER_TIMEOUT_MS` | `120000` | Compiler request timeout in milliseconds |

## Documentation

Refer to the following documentation for more information on using and developing FlagCV:

- [Architecture](docs/architecture.md) — Runtime design, kernel compilation, and multi-backend support.
- [Operator Inventory](docs/operators.md) — Supported operators and hardware platforms.
- [Operator Development](docs/operator-development.md) — Adding and optimizing operators.
- [Testing](docs/testing.md) — Correctness testing and backend validation.
- [Benchmarking](docs/benchmark.md) — Performance testing and comparison methods.

## Contribution

Contributions to FlagCV are welcome. Please refer to the [Contribution Guide](CONTRIBUTING.md) for development guidelines.

Please file an issue for feature requests or bug reports.

## License

FlagCV is licensed under the [Apache License (Version 2.0)](LICENSE).
