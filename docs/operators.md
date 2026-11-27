# FlagCV 算子与平台支持

C++ 目标由调用方预分配，与源图像同形状、同深度，支持 1–4 个交错通道和独立
字节 pitch；允许精确原地别名及同一显存分配内不相交的 ROI，不允许部分像素重叠。
可选 mask 为同尺寸单通道 8U，零值处保留原目标像素。内部 Python Core 使用连续
单通道输入，不代表完整功能矩阵。

| 算子 | C++ 语义范围 | 内部 Python Core 范围 |
| --- | --- | --- |
| `add` | 8U/16U 饱和，16F/32F；可选 mask | 8U/32F |
| `subtract` | 8U/16U 饱和，16F/32F；可选 mask | 8U/32F |
| `multiply` | 8U/16U 饱和，16F/32F；默认单位缩放，显式有限 `scale` 仅支持 32F | 8U/32F，单位缩放 |
| `bitwise_and` | 全部 `Depth`，含浮点存储位；可选 mask | 8U |
| `bitwise_not` | 全部 `Depth`，含浮点存储位；可选 mask | 8U |
| `flip` | 全部 `Depth`，逐位搬运；正值反转列、零反转行、负值同时反转 | 8U/32F，三种方向 |

16U multiply 的原生性能对照使用乘积落在 int32 范围内的共同输入域。参考例外
和数值规则见[数值验收](numerical-tolerance.md)，接口见 [C++ 指南](cpp-operators.md)。

## 平台支持范围

| 后端 / 设备 | 工具链与状态 |
| --- | --- |
| `amd` / W7900D | ROCm 7.2.4、FlagTree `0.7.0+amd3.6`、`gfx1100` wave32；已验证功能与树外安装，未启用 ATen 原生比较。 |
| `hygon` / BW1000 | DTK 26.04、FlagTree `0.7.0+hcu3.6`、`gfx936` wave64；已验证功能与树外安装，ATen 仅在[限定 Python 域](hip-validation.md#python-原生比较覆盖)可用。 |
| `hygon` / BW1101 | DTK 26.10、FlagTree `0.7.1a1+hcu3.6`、`gfx938` wave64；已验证功能与树外安装，ATen 原生比较待验证、未启用。 |
| `nvidia` / H20 | 已完成固定 FlagTree 的 SM90 离线编译及 PTX Scratch ABI 检查；该工具链的实卡 launch、迁移安装、原生采集和新协议性能对照待完成。H800 是 collector 注册默认，不代表实测。 |
| `iluvatar` | Preview：提供 CoreX 后端，原生执行待验证。 |
| `mthreads` | Preview：提供 MUSA 后端，原生执行待验证。 |
| `thead` | Preview：提供 PPU 后端，原生执行待验证。 |
| `ascend` | Planned：保留后端源码，C++ 构建不启用。 |

AMD/Hygon 功能范围绑定[历史验收所列候选与工具链](https://github.com/qcxl99/FlagCV/blob/adc01fda1014a530f2a844ad28746829e46a2da0/docs/hip-validation.md#六算子整改验证)：
双卡公开 C++ API（含便利重载）、外部原生 stream、SDK 设备状态恢复、内部 Python
和迁移后树外消费；功能验证不等于已具备性能基线。H20 的[旧协议记录](baselines/nvidia-h20-2026-11-25/README.md)
不替代固定 FlagTree 的设备验证。[AMD/Hygon 指南](hip-validation.md) · [NVIDIA 指南](nvidia-validation.md)。

### Preview 和 Planned 资格

Preview 后端应核对实际编译器、产物参数布局及 scratch 语义，并完成设备上的公开
API 验证；仅能构建插件不是原生执行资格。

Ascend [构建规则](../backends/ascend/CMakeLists.txt)拒绝启用；[预留源码](../backends/ascend/backend.cpp)
拒绝创建 Handle，不初始化 AscendCL、不切设备、不获取设备资源。重新开放前，
验证目标设备的 CANN/FlagTree 产物 ABI（含打包参数和 scratch 约定），以及两个
Handle 借用同一调用方 context、buffer 和原生 stream 的生命周期。释放一个 Handle
只清理其拥有的模块登记，不破坏另一 Handle、调用方资源或在途工作。验证 context/
device 错配错误；device reset 与 AscendCL finalization 留给应用程序。
