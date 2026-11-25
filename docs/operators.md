# FlagCV 算子清单

现有 Core Python 评测使用连续单通道输入。C++ 使用调用方预分配的目标视图，
各算子规定其尺寸、深度和通道。各 ROI 可以位于同一显存分配中的不相交区域。
本表的维护按算子族分配，新增 PR 在清单中登记负责人。

| 算子 | C++ 语义范围 | Python 范围 | NVIDIA | 维护 |
| --- | --- | --- | --- | --- |
| `add` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 算术维护者 |
| `subtract` | 8U/16U 饱和，16F/32F；可选 mask，允许原地 | 8U/32F | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 算术维护者 |
| `multiply` | 8U/16U 饱和，16F/32F；默认单位缩放，32F 支持显式有限缩放，允许原地 | 8U/32F，单位缩放 | 公开 API、Python、OpenCV CUDA 逐例基线，16U 使用共同数值域测速 | Core 算术维护者 |
| `bitwise_and` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 位运算维护者 |
| `bitwise_not` | 全部 `Depth`，含浮点存储位；可选 mask，允许原地 | 8U | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 位运算维护者 |
| `flip` | 全部 `Depth`，逐位搬运；正、零、负方向，允许原地 | 8U/32F，三种方向 | 公开 API、Python、OpenCV CUDA 逐例基线 | Core 几何维护者 |
| `cvtColor` | 84 个 8U、20 个 16U、20 个 32F code-depth 组合，通道、视频尺寸与拒绝继承 contrib | 原四种 8U 已纳入，共用 kernel | H20 公开 API 正确性，性能验收待补 | Imgproc 颜色维护者 |
| `resize` | 8U/32F C1/C3/C4 单图，8U C1/C3/C4 批量，四种插值，比例或指定尺寸，预备表与独立 pitch | 性能验收待补 | H20 公开 API 正确性，性能验收待补 | Imgproc 几何维护者 |
| `warpAffine` | 8U/16U/32F，C1/C3/C4，独立 pitch 与 ROI；目标尺寸取自 dst；正向或逆向 2×3 矩阵；禁止重叠 | 独立 Python 性能验收待补 | H20 公开 API 正确性切片；性能验收待补 | Imgproc 几何维护者 |

`warpAffine` 默认 LINEAR 和零常量边框。支持 NEAREST、LINEAR、CUBIC 和
AREA（LINEAR 别名）。边框支持 CONSTANT、REPLICATE、REFLECT、REFLECT_101、
WRAP、TRANSPARENT。透明 LINEAR 以已初始化目标像素补足缺失的插值 taps。
透明 NEAREST 与 LINEAR 保留完全在源图之外的目标像素。CUBIC + TRANSPARENT
会拒绝。矩阵及边框必须有限且处于所选运算的索引范围。
数值参考与固定 CPU 配置见 [数值验收](numerical-tolerance.md)。

CUDA 逐例基线覆盖已核对的共同支持范围。16F、C++ 掩码、原地操作、非单位
缩放和其他位运算深度的本轮证据见功能矩阵。并行推进算子时按 PR 登记具体负责人。
本轮逐例结果和环境见 [NVIDIA 验收](nvidia-validation.md)。

共享 HIP 后端已有 Core 算子的公开接口和数值标准。W7900D（gfx1100）
与 BW1000（gfx936）有单卡公开 API 和 Python 正确性证据。BW1101
（gfx938）有构建及离线 ABI 证据，设备验收仍为预览。新增的 warpAffine 正在按同一源码开展真机验收。按
[HIP 验收](hip-validation.md) 的平台范围维护支持状态。

## 后续批次

| 顺序 | 范围 | 状态 |
| --- | --- | --- |
| 1 | `bitwise_or`、`bitwise_xor`、`compare`、`min`、`max` | OR/XOR/min/max 有独立评测 PR 提案；公开 API 和验收待接入 |
| 2 | `warpPerspective`、`remap` | 优先复用已有 Triton 和 OpenCV 工作；按插值及边界模式建立 CPU 和 CUDA 基线 |
| 3 | `GaussianBlur`、`filter2D` | 按边界策略和采样规则建立参考样本 |

AMD 新工具链与新设备继续按 HIP 交付流程验证。[AMD PR #3](https://github.com/flagos-ai/FlagCV/pull/3)

## 边框与阈值

| 算子 | C++ 语义范围 | 验证入口 |
| --- | --- | --- |
| `copyMakeBorder` | C1–C4 和已有 `Depth`。支持 CONSTANT、REPLICATE、REFLECT、REFLECT_101、WRAP。默认按孤立视图处理，显式父 ROI 可借用父图像素。 | `flagcv.{hip,nvidia}.border_threshold` 和 `tools/check_border_threshold.py` |
| `threshold` | 五种固定模式，8U/16U/16S/32F/64F。Otsu 支持 U8/U16 C1，Triangle 支持 U8 C1。自动模式提供设备结果与返回主机值的同步入口。 | 同上，含独立 OpenCV CPU 及阈值返回值检查 |

主题的设备和性能证据按实际平台与执行入口提交。FP32/FP64 参考采用固定 SSE3 OpenCV。
