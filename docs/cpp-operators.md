# C++ operator interfaces

The C++ API is declared in [core.hpp](../include/flagcv/core.hpp) and exported through
`<flagcv/flagcv.hpp>`. See the declarations for each operator's argument order,
layout requirements and numerical semantics.

## Operands and streams

`GpuMat` is a non-owning view of caller-allocated device memory. Inputs and
destination must satisfy the operator's extent, depth and channel requirements.
Each operand may have an independent byte pitch and a base offset aligned to
its element size. Lowering validates layout and operand compatibility before compilation
or launch.

The listed operators accept exact source/destination aliasing and disjoint views.
Lowering rejects partial pixel overlap. Add, subtract, AND and NOT accept an
8U single-channel mask and preserve existing destination pixels wherever it
is zero; initialize the destination before a masked call. Multiply defaults to
scale 1; only 32F accepts a finite non-unit scale representable within the 32F range.
See the [operator inventory](operators.md) for the supported depth/channel matrix.

Calls enqueue on the supplied native stream (default `nullptr`). The caller
owns the stream and buffers in a compatible device context and synchronizes
before reading results or releasing storage. The caller retains the Handle until the stream has
finished its enqueued FlagCV work. One Handle binds one backend and device.
Use distinct Handles for simultaneous calls on different host threads.

## Explicit and convenience calls

These examples assume `src` and `dst` are views of caller-managed CUDA buffers
on device 0 and `stream` is a caller-created `cudaStream_t` for the same device
and primary context. Choose either form:

```cpp
flagcv::Handle handle("nvidia", 0);
flagcv::flip(handle, src, dst, 1, stream);
cudaStreamSynchronize(stream);
```

```cpp
flagcv::setDevice("nvidia", 0);
flagcv::flip(src, dst, 1, stream);
cudaStreamSynchronize(stream);
```

`setDevice` selects the backend and device for subsequent convenience calls
on this thread. It does not set the SDK current device or affect explicit
Handles; a failed selection leaves the previous selection intact. A successful
selection is required before convenience calls. The thread retains a Handle
and its loaded modules for each selected pair until thread exit. Complete work
before that exit (or before destroying an explicit Handle), then release the
stream and buffers yourself.

## Call path

Convenience overloads forward to the explicit overloads with the selected
Handle. Both call family lowering, then the shared runtime. The runtime
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
