# FlagCV 算子与平台支持

Python 评测当前为连续单通道输入。C++ 使用预分配同形状、同深度的目标，支持
1–4 个交错通道和独立字节 pitch。各 ROI 可以位于同一显存分配中的不相交区域。
本表的维护按算子族分配，新增 PR 在清单中登记负责人。

| 算子 | C++ 语义范围 | 内部 Python Core 范围 | 维护 |
| --- | --- | --- | --- |
| `add` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | Core 算术维护者 |
| `subtract` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | Core 算术维护者 |
| `multiply` | 8U/16U 饱和，16F/32F；默认单位缩放，32F 支持显式有限缩放，允许原地 | 8U/32F，单位缩放 | Core 算术维护者 |
| `bitwise_and` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | Core 位运算维护者 |
| `bitwise_not` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | Core 位运算维护者 |
| `flip` | 全部 `Depth`，逐位搬运；正、零、负方向，允许原地 | 8U/32F，三种方向 | Core 几何维护者 |

原生比较使用共同输入域，16U multiply 的输入上界保证乘积位于 int32
范围内。内部 Python Core 只列生产 kernel 的主要计时子集。功能测试另行
覆盖 16F、mask、原地、非单位缩放及其他位运算深度。

## 平台支持范围

| 平台 | 验证范围 |
| --- | --- |
| Hygon BW1000 | 双卡公开 API 与原生 Stream、内部 Python、迁移安装、Core 逐例性能与分派。 |
| NVIDIA H20 | FlagTree SM90 离线编译与 scratch ABI。实卡和新性能采集待复验。 |
| Hygon BW1101 | `fd6153c` 双卡公开 API、便利重载、原生 Stream、设备恢复、Python 与迁移安装通过。Core 延迟为共享卡观测。 |
| AMD W7900D | `fd6153c` 双卡公开 API、便利重载、原生 Stream、设备恢复、Python 与迁移安装通过。Core 保留实测延迟。 |
| Iluvatar（`iluvatar`） | 提供 CoreX backend，原生执行待资格验证。 |
| MThreads（`mthreads`） | 提供 MUSA backend，原生执行待资格验证。 |
| T-Head（`thead`） | 提供 PPU backend，原生执行待资格验证。 |
| Ascend（`ascend`） | 保留后端源码，当前构建不启用。 |

平台工具链与历史支持见 [NVIDIA 验收](nvidia-validation.md) 和
[HIP 验收](hip-validation.md)。

新增工具链或设备按对应平台流程验证后登记。新增算子及原生对照的接入方式见 [算子开发指南](operator-development.md)。
