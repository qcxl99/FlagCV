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
owns the stream and device buffers on the Handle's device and synchronizes
before reading results
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


## warpAffine

```cpp
const flagcv::AffineMatrix matrix{1.0, 0.0, 12.0, 0.0, 1.0, -4.5};
flagcv::warpAffine(handle, src, dst, matrix, flagcv::Interpolation::kLinear,
                   flagcv::Border::kConstant, {}, false, stream);
```

调用方按所需输出尺寸预分配 `dst`。矩阵按行给出，`inverse_map` 为真时直接
从目标映射到源坐标。调用方保留句柄、视图存储和原生 stream，直到提交完成。
仿射路径使用调用方给定的图像存储，暖态每次调用提交一个 kernel。


## Resize

`ResizePlan` 在 CPU 解析尺寸并准备系数。构造时提供源尺寸、深度、通道和
`ResizeOptions`。调用方按 `destination_size()` 分配目标。单图中两个正
尺寸优先，任一非正时用有限正 `fx/fy` 派生。批量只接受两个正尺寸或
`{0,0}` 加两个比例，支持等形的 8U C1/C3/C4，输入和输出各自使用字节
batch pitch。每幅图可以使用独立于 packed width 的行 pitch。

有系数的计划需要把 `table_data()` 的 `table_bytes()` 字节上传到调用方
分配的设备区域，并保持该区域只读，直到所有使用流完成。调用方负责让上传对
每条流可见。相同计划和设备副本可直接重复使用。零字节路径传空 workspace。
`resize` 和 `resizeBatch` 仅提交设备工作，调用方保留句柄、图像和工作区。


## cvtColor

使用 OpenCV 的数值转换码。`cvtColorOutput()` 返回分配目标所需的尺寸和
通道，深度与源相同。NV12/NV21 的单缓冲以 W×3H/2 输入，返回 W×H。
有色 4:2:2 解码需要偶宽，灰度抽取可用奇宽。输入和输出的像素区域必须不重叠。

使用 `cvtColorTableKind(code)` 获取表族。对非空表族，用
`cvtColorTableData()` 取得只读宿主数据，上传后把设备地址和容量传给
`cvtColor()`。同表族的转换可共享同一设备副本。调用方必须上传该表族的
准确内容，并负责流之间的可见性和完成前的寿命。Lab 和逆向 Luv 需要
8 字节对齐，其他表族需要 4 字节。无表族用空 workspace。
