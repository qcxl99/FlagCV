# C++ operator interfaces

The C++ API is declared in `include/flagcv/core.hpp` and exported through
`<flagcv/flagcv.hpp>`. See the declarations for each operator's argument order,
layout requirements and numerical semantics.

## Operands and streams

`GpuMat` is a non-owning view of caller-allocated device memory. Inputs and
destination must satisfy the operator's extent, depth and channel requirements.
Each operand may have an independent byte pitch and a base offset aligned to
its element size. Lowering validates these requirements before compilation
or launch.

All six operators accept exact source/destination aliasing and disjoint views.
Lowering rejects partial pixel overlap. Add, subtract, AND and NOT accept an
8U single-channel mask and preserve existing destination pixels wherever it
is zero. Multiply defaults to scale 1 and accepts finite explicit 32F scale.
The supported depth/channel matrix is recorded in the operator inventory.

Calls enqueue on the supplied native stream (default `nullptr`). The caller
owns the stream and device buffers and synchronizes before reading results
or releasing storage. The caller retains the Handle until the stream has
finished its enqueued FlagCV work. One Handle binds one backend and device.
Use distinct Handles for simultaneous calls on different host threads.

## Call path

The entry points call family lowering, then the shared runtime. The runtime
resolves the platform registry and the common `kernels/registry.json`, asks the external compiler process for a
binary on a cache miss, and launches through the backend plugin.

Pointwise operations share Triton kernels for packed, pitched and masked images.
Sizes and strides remain runtime arguments, with alignment and divisibility
specializations, so compatible layouts reuse compiled binaries. Platform
providers choose the launch plan; backend plugins load the artifact and
enqueue execution on the caller's stream.

See [Architecture](architecture.md) for the compiler protocol, cache and plugin
boundary, and [Operator development](operator-development.md) for the steps to connect a
Triton operator to a C++ interface, lowering, kernel registry, compilation
plan and backend launch.
