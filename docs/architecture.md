# FlagCV Architecture

FlagCV is an image processing library with OpenCV-shaped interfaces and
FlagDNN-shaped multi-backend plumbing. The public API immediately enqueues one operator on a caller-owned native
stream. The runtime invokes compilation on a cache miss.

## Layers

```
include/flagcv/        public C++ API (OpenCV-shaped)
      |
src/core, src/imgproc  one source file per operator family
      |
src/lowering           operands + params -> LoweredOp (validation lives here)
      |
src/runtime            cache lookup, compile on miss, launch
      |
compiler/              out-of-process Triton compiler (Python)
      |
backends/<platform>/   dlopen'd plugin: context, module load, launch
```

Each arrow is one-way. The core library links no vendor SDK and no Python; it
reaches the toolchain only by talking to the external compiler process, and
reaches the device only through a backend plugin.

## Kernel resolution

Two registries are consulted, in order:

1. `backends/<platform>/kernels/registry.json`
2. `kernels/registry.json` (common Triton kernels)

Platform registration is **exclusive**. Once a platform claims an operator,
a failure there is reported to the caller — the common kernel is never silently
tried as a fallback. So every constraint a platform kernel imposes (depth,
channel count, alignment) must be checked in that platform's `compiler.py`
before compilation, not discovered during it.

## Compile and launch path

```
flagcv::flip(handle, src, dst, code)
  lower_flip()                       validate, build LoweredOp
  launch()                           cache key -> Handle's kernel cache
    miss: resolve_kernel()           platform registry, then common
          CompilerClient::request()  JSON line to the compiler process
            python3 -m flagcv_codegen.main     (started once, kept alive)
              backends/<platform>/compiler.py  choose blocks/warps/grid
              triton.compile(target from the plugin)
              -> artifact file + launch metadata
          plugin.load(artifact bytes)
    plugin.launch(grid, args, stream)  no compiling, no synchronizing
```

The compiler process is started on the first cache miss and serves every
`Handle` in the host process, so the Python and Triton import is paid once
per process rather than per compile. It never initialises a device: the
plugin reports the compile target (`cuda:90:32`), so the compiler imports no
framework and needs no GPU. The protocol is one JSON object per line; it is
documented in `compiler/flagcv_codegen/main.py`.

Each compiler exchange uses a monotonic deadline for writing and reading.
`FLAGCV_COMPILER_TIMEOUT_MS` defaults to 120000 ms. A timeout discards the
protocol connection, terminates the compiler's process group and attempts
bounded reaping. A later request can start a fresh compiler after reaping.
Closing a Handle releases its loaded modules. At process shutdown the shared
compiler gets a short EOF grace period before termination.

### Selecting a device

`flagcv::setDevice(backend, ordinal)` selects the Handle for convenience
overloads on the calling thread. Each thread keeps one Handle and its loaded
modules per backend and ordinal until thread exit. A failed selection
preserves the thread's previous selection. Convenience calls require a
successful selection first. Calls with an explicit Handle are independent.

Device selection preserves the calling thread's SDK device and context.
Buffers and native streams must belong to a context compatible with the
selected Handle. NVIDIA uses the device's primary context. Keep buffers and
streams alive and complete queued work before the owning thread exits.

**Loaded modules.** Each Handle owns a cache in the context of its bound
device. It matches the kernel name and operand/parameter signature, including
depth, channels, 16-byte base alignment, constexpr values and runtime argument
types. Runtime image sizes and strides can reuse a module when their types
and any `add_hinted_param` divisibility classes match. The Runtime queries
compiler identity once per backend per process. Restart the process after
changing registered kernels, providers or compiler settings.

**Files.** Compiled binaries go to `$FLAGCV_CACHE_DIR/artifacts/<hash>/`
(default `~/.cache/flagcv`), keyed by Triton's compiled artifact hash.
The hash reflects target, compiler, options and signature. Triton's own cache
(`~/.triton/cache`) holds intermediate IR. A new process starts a new compiler
process and may reuse Triton artifacts. Each Handle still loads its own device
modules and releases them before the context.

**Locations and overrides.** `FLAGCV_PYTHON` selects the interpreter for the
compiler process (default `python3`). Kernels, the compiler and platform
providers are found under `FLAGCV_DATA_DIR`, else the installed data directory
relative to libflagcv, else the source tree it was built from. Plugins are found under
`FLAGCV_PLUGIN_DIR`, else `flagcv/` next to libflagcv, else the build tree.

## Scratch ABI qualification

The NVIDIA provider checks the pinned FlagTree NVIDIA installation and its
Triton 3.6.0 module path. The compiler checks that both scratch allocation sizes are present and
zero. It also verifies the emitted PTX entry layout against the Runtime
signature, including the two trailing 64-bit pointer parameters. The Runtime
compares that count with the backend expectation before loading the binary.

The AMD and Hygon plugins accept separate bounded Triton 3.6.0 compiler/target sets
(gfx936, gfx938 and gfx1100). Both qualified compiler families emit two
trailing pointer slots. FlagCV validates their LLVM entry signature and
checks that both pointers have the `readnone` attribute. The profile scratch
size must be explicitly zero. Some HIP compilers omit the global size, so
the unused-pointer proof is required per artifact. An explicit nonzero size
is rejected. Platform delivery status also requires on-device correctness
and installation evidence.

Iluvatar, MThreads, T-Head and Ascend need their exact compiler build,
emitted argument layout, scratch metadata semantics and an on-device public API
test before enabling native launch. Ascend is excluded from C++ builds until
its device lifecycle is qualified. Its source rejects Handle creation without
initializing AscendCL, changing the active device or acquiring device resources.
Internal Python evaluation compiles through the same provider, registry and
specialization. Its launcher is supplied by the selected FlagTree distribution.
Current NVIDIA offline and device status is recorded in [NVIDIA validation](nvidia-validation.md).

The process-group cleanup follows the approach in
[FlagDNN's compiler client](https://github.com/flagos-ai/FlagDNN/blob/a84bcc18a172fa01597d2fc171ce3af680f3074a/src/runtime/compiler_client.cpp#L671).
CUDA validation checks the PTX emitted by the installed FlagTree compiler.

## Ascend enablement

Before enabling the backend, verify that two Handles can use the caller's
existing context, buffers and native stream. Check that destroying either
Handle preserves the other Handle, the caller's context and stream, and
ongoing work. FlagCV must release only its owned module registrations while
the application retains device reset and AscendCL finalization. Verify the
error path for a mismatched context or device and the supported CANN/FlagTree
artifact ABI on the target device.

## Units: pitch vs stride

The public API takes `row_pitch` in **bytes**, because imaging hardware pads
rows for DMA and that padding is independent of `width * channels *
depth_size`. Kernel pointers are typed by pixel depth, so their arithmetic
advances in **elements**.

`GpuMat::row_stride()` is the single place that conversion happens. Everything
below the public API — `ImageSpec::row_stride`, the params a `LoweredOp`
carries, kernel arguments — is in elements.

## Masks

Where the mask is zero, the destination keeps its prior contents byte-identical. Kernels implement this by
folding the mask into the store predicate, never with
`tl.where(mask, result, 0)`. A masked call therefore requires an initialized
`dst`.

## Memory ownership

FlagCV never allocates or frees device memory. `GpuMat` is a non-owning view
over a buffer the caller owns — unlike `cv::cuda::GpuMat`, which owns its
data. The caller keeps all buffers, the stream and the Handle alive until the enqueued work completes.

## Plugin boundary

Plugins use the ABI in `backends/backend.hpp` and may share internal helpers
in `backends/common`. They receive operand layouts and params as plain data
and do not include core internal headers.

### Backends

| `FLAGCV_BACKENDS` | SDK | Artifact | Warp | Note |
|---|---|---|---|---|
| `nvidia` | `cuda.h` | `.cubin` | 32 | |
| `iluvatar` | CoreX `cuda.h` | `.cubin` | 64 | needs `CUDAToolkit_ROOT` pointed at CoreX |
| `amd` | `hip/hip_runtime.h` | `.hsaco` | 32 | AMD ROCm |
| `hygon` | `hip/hip_runtime.h` | `.hsaco` | 64 | Hygon DTK |
| `mthreads` | `musa.h` | `.mubin` | 32 | Triton backend name is `mtgpu` |
| `thead` | `hggc.h` | `.hgbin` | 32 | Triton backend name is `PPU` |
| `ascend` (unavailable) | `acl/acl.h`, `rt.h` | `.npubin` | 1 | preview source, excluded from C++ builds |

The module-based plugins share `backends/common/driver_backend.hpp`:
their driver APIs have the CUDA shape (init, primary context, push/pop, load
module from bytes, get function, launch with an argument array), so a plugin
is just an `Api` struct naming its own calls. Ascend does not have that shape
— it registers the binary with the runtime and passes a packed argument
buffer — so it implements the table itself and sets
`appends_scratch_args = 0` in the preview code. This is a provisional ABI
expectation and must be verified against the target CANN/FlagTree build before
enabling Ascend.

AMD and Hygon share the internal HIP driver and thread-local device binding.
Their plugins resolve the SDK independently in separate build trees. Their
backend names, providers, registries and published artifact paths are distinct.

The plugin also reports the compile target (`cuda:90:32`, `hip:gfx936:64`,
...) and the Triton backend name, which is what lets the compiler process
build for a device it never touches. Each platform provider defines its artifact format, Triton backend name,
element-block limits and artifact validation.

The [operator inventory](operators.md) and [HIP validation](hip-validation.md)
track which device/toolchain combinations have native evidence. Plugin
buildability and offline ABI checks are recorded separately.

The runtime and provider split follows the FlagDNN approach. Python operator
evaluation retains the case, candidate and JSON interfaces used by FlagGems.
[FlagDNN](https://github.com/flagos-ai/FlagDNN) ·
[FlagGems](https://github.com/flagos-ai/FlagGems)
