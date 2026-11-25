# FlagCV 算子清单

Python 评测当前为连续单通道输入。C++ 使用预分配同形状、同深度的目标，支持
1–4 个交错通道和独立字节 pitch。各 ROI 可以位于同一显存分配中的不相交区域。
本表的维护按算子族分配，新增 PR 在清单中登记负责人。

| 算子 | C++ 语义范围 | Python 范围 | NVIDIA | 维护 |
| --- | --- | --- | --- | --- |
| `add` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 算术维护者 |
| `subtract` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 算术维护者 |
| `multiply` | 8U/16U 饱和，16F/32F；默认单位缩放，32F 支持显式有限缩放，允许原地 | 8U/32F，单位缩放 | 公开 API、Python、OpenCV CUDA 逐例基线，16U 使用共同数值域测速 | Core 算术维护者 |
| `bitwise_and` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 位运算维护者 |
| `bitwise_not` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 位运算维护者 |
| `flip` | 全部 `Depth`，逐位搬运；正、零、负方向，允许原地 | 8U/32F，三种方向 | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 几何维护者 |

CUDA 逐例基线覆盖已核对的共同支持范围。16F、C++ 掩码、原地操作、非单位
缩放和其他位运算深度的本轮证据见功能矩阵。并行推进算子时按 PR 登记具体负责人。
本轮逐例结果和环境见 [NVIDIA 验收](nvidia-validation.md)。

共享 HIP 后端沿用上表算子的公开接口和数值标准。W7900D（gfx1100）
与 BW1000（gfx936）有单卡公开 API 和 Python 正确性证据。BW1101
（gfx938）有构建及离线 ABI 证据，设备验收仍为预览。按
[HIP 验收](hip-validation.md) 的平台范围维护支持状态。

## 后续批次

| 顺序 | 范围 | 状态 |
| --- | --- | --- |
| 1 | `bitwise_or`、`bitwise_xor`、`compare`、`min`、`max` | OR/XOR/min/max 有独立评测 PR 提案；公开 API 和验收待接入 |
| 2 | `resize`、`warpAffine`、`warpPerspective`、`remap` | 优先复用已有 Triton 和 OpenCV 工作；按插值及边界模式建立 CPU 和 CUDA 基线 |
| 3 | `cvtColor`、`GaussianBlur`、`filter2D`、`threshold` | 先登记支持的转换代码、通道、边界策略和参考样本；`cvtColor` 为性能重点 |

AMD 新工具链与新设备继续按 HIP 交付流程验证。[AMD PR #3](https://github.com/flagos-ai/FlagCV/pull/3)
