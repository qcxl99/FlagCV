# FlagCV Architecture

FlagCV exposes OpenCV-style C++ APIs and uses caller-managed image buffers.
Each call submits an operator to a caller-owned native stream. The runtime
invokes compilation on a cache miss.

## Layers

```text
include/flagcv/        public C++ API (OpenCV-style)
      |
src/core/              operator-family entry points
      |
src/lowering           operands + params -> LoweredOp (validation lives here)
      |
src/runtime            cache lookup, compile on miss, launch
      |-- compiler/             out-of-process Triton compiler (Python)
      `-- backends/<platform>/  dlopen'd plugin: context, module load, launch
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

```text
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

**Files.** Compiled binaries go to `$FLAGCV_CACHE_DIR/artifacts/<backend>/<hash>/`
(default cache root `~/.cache/flagcv`), with a separate namespace per FlagCV
backend and a directory keyed by Triton's compiled artifact hash.
The hash reflects the target, compiler, options, signature and JIT dependencies.
Triton's own cache (`~/.triton/cache`) holds intermediate IR. A new process starts a new compiler
process and may reuse Triton artifacts. Each Handle still loads its own device
modules and releases them before the context.

**Locations and overrides.** `FLAGCV_PYTHON` selects the interpreter for the
compiler process (default `python3`). Kernels, the compiler and platform
providers are found under `FLAGCV_DATA_DIR`, else the installed data directory
relative to libflagcv, else the source tree it was built from. Plugins are found under
`FLAGCV_PLUGIN_DIR`, else `flagcv/` next to libflagcv, else the build tree.

## Scratch ABI qualification

FlagCV does not provision kernel scratch buffers. A qualified provider must
verify the artifact's argument layout and show that no scratch storage is
required. The compiler reports the verified trailing scratch argument count;
the runtime rejects a mismatch with the plugin's ABI before loading the
artifact. The runtime passes null pointers for the qualified unused slots.
Internal Python evaluation shares registry, specialization and provider
validation, but launches through the selected FlagTree distribution.

See the [NVIDIA](nvidia-validation.md#scratch-abi) and
[AMD/Hygon](hip-validation.md#scratch-abi) artifact checks and the
[platform inventory](operators.md#平台支持范围) for qualification status.

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

`flagcv::GpuMat` is a non-owning view; FlagCV does not allocate or free the
caller's image buffers. This is an ownership contract for FlagCV, not a claim
about OpenCV's `cv::cuda::GpuMat`, which can also wrap external memory.
Keep all buffers, the stream and the Handle alive until queued work completes.
Use separate Handles for simultaneous calls from different host threads.
See [C++ operator interfaces](cpp-operators.md) for overlap and operand rules.

## Plugin boundary

Plugins use the ABI in `backends/backend.hpp` and may share internal helpers
in `backends/common`. They receive operand layouts and params as plain data
and do not include core internal headers.

Module-based drivers can share [driver helpers](../backends/common/driver_backend.hpp)
for context binding, module loading and launch. AMD and Hygon share internal
HIP driver helpers, but resolve SDKs independently and keep separate backend
names, providers, registries and artifact paths.

The plugin also reports the compile target (`cuda:90:32`, `hip:gfx936:64`,
...) and the Triton backend name, which is what lets the compiler process
build for a device it never touches. Each platform provider defines its artifact format, Triton backend name,
element-block limits and artifact validation.

The [platform inventory](operators.md#平台支持范围) distinguishes plugin
availability, offline ABI checks and device evidence. See the [backend source](../backends)
for the actual SDK and provider definitions.

The runtime and provider split follows the FlagDNN approach. Python operator
evaluation retains the case, candidate and JSON interfaces used by FlagGems.
[FlagDNN](https://github.com/flagos-ai/FlagDNN) (including the
[compiler-client cleanup](https://github.com/flagos-ai/FlagDNN/blob/a84bcc18a172fa01597d2fc171ce3af680f3074a/src/runtime/compiler_client.cpp#L671)) ·
[FlagGems](https://github.com/flagos-ai/FlagGems)
