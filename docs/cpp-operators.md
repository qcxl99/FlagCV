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

### copyMakeBorder

`copyMakeBorder` 接收预分配目标和非负的 `BorderExtents`。
普通视图按 `BORDER_ISOLATED` 处理。需要 OpenCV 的父图像素时，传入
`BorderParent{parent, x, y}`，其中坐标是源视图在同一 pitched 父图内的原点。
运行时核对指针、类型、pitch 和范围，按每条边的可用像素扩展视图，再检查扩展视图
和目标是否重叠。其余像素按所选边界计算。零边距且源目标完全相同可直接返回。
常量按通道转换为目标深度。读取源像素的路径保留存储位，包括浮点的符号和载荷。
本入口支持 C1–C4，拒绝 TRANSPARENT。

### threshold

`threshold` 返回固定阈值。整数模式返回向下取整的值，要求取整结果能表示为
int32。浮点模式返回原 double。像素等于阈值时属于小于等于分支。
`maxval` 对整数先按固定 CPU 的 int32 取偶舍入，然后按输出深度饱和。
16S TRUNC 低于 `-32768` 时，返回原取整阈值并将像素置零。
固定模式支持完全原地，拒绝部分重叠。dry-run 不需要目标，也不提交设备工作。

`thresholdAuto` 使用调用方分配的 `ThresholdScratch{void*, bytes}`。
用 `thresholdWorkspaceBytes` 查询最小尺寸，起始指针对齐到 int32。
第一个 int32 保存有符号结果，后面是暂存直方图。调用方在流完成前独占这片可写内存，
并与活跃源和目标字节范围分离。即使后续编译、提交或读回抛错，也必须保留之前已提交
工作使用的 Handle、流和缓冲区直到完成。只读 `DeviceWorkspace` 继续用于上传表。
自动 dry-run 仍提交直方图和阈值选择，省去目标 Kernel。

`thresholdAutoHost` 在提交前检查 readback 能力。它等待所选原生流，然后进行
厂商提供的阻塞 D2H。整个等待和 D2H 都属于公开调用耗时。原生阻塞复制可能带来的
其他等待按设备实测记录。HIP 和 NVIDIA 提供该扩展，公开 Handle 和插件 V2 表
保持原样。
