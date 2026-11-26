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
| `cvtColor` | 84 个 8U、20 个 16U、20 个 32F code-depth 组合，通道、视频尺寸与拒绝继承 contrib | 已安装 Triton JIT 覆盖共享 code-depth 路径 | H20 公开 C++、Python JIT 及同卡配对与单端数据 | Imgproc 颜色维护者 |
| `resize` | 8U/32F C1/C3/C4 单图，8U C1/C3/C4 批量，四种插值，比例或指定尺寸，预备表与独立 pitch | 已安装 Triton JIT 支持单图和批量 | H20 公开 C++、Python JIT 及单图与独立批量数据 | Imgproc 几何维护者 |
| `warpAffine` | 8U/16U/32F，C1/C3/C4，独立 pitch 与 ROI；目标尺寸取自 dst；正向或逆向 2×3 矩阵；禁止重叠 | 已安装 Triton JIT，保持相同参数与分派 | H20 公开 C++、Python JIT，逐例可比与不可比结果分列 | Imgproc 几何维护者 |

`warpAffine` 默认 LINEAR 和零常量边框。支持 NEAREST、LINEAR、CUBIC 和
AREA（LINEAR 别名）。边框支持 CONSTANT、REPLICATE、REFLECT、REFLECT_101、
WRAP、TRANSPARENT。透明 LINEAR 以已初始化目标像素补足缺失的插值 taps。
透明 NEAREST 与 LINEAR 保留完全在源图之外的目标像素。CUBIC + TRANSPARENT
会拒绝。矩阵及边框必须有限且处于所选运算的索引范围。
数值参考与固定 CPU 配置见 [数值验收](numerical-tolerance.md)。

OpenCV CUDA 逐例对照覆盖已核对的共同支持范围。16F、C++ 掩码、原地操作、非单位
缩放和其他位运算深度的本轮证据见功能矩阵。并行推进算子时按 PR 登记具体负责人。
本轮逐例结果和环境见 [NVIDIA 验收](nvidia-validation.md)。

## 透视与重映射

| 算子 | 已实现的 C++ 入口和范围 |
| --- | --- |
| `warpPerspective` | `flagcv::warpPerspective`，支持正向或逆向 3×3 矩阵 |
| `remap` | `flagcv::remap`，支持交错或双图浮点坐标、定点图和相对坐标 |

两项均通过 `<flagcv/imgproc.hpp>` 公开，支持 8U/16U/32F 的 C1/C3/C4、
独立行距与 ROI、NEAREST、LINEAR 和 CUBIC。AREA 使用 LINEAR。支持六种
边框，透明边框要求预先初始化目标。具体坐标图格式、尺寸和拒绝条件见
[C++ 接口](cpp-operators.md#warpperspective-and-remap)。

H20 已在迁移安装包上验证公开 C++ 与 Python kernel。C++ 验证入口为
`flagcv.nvidia.projective`，以及树外通过 `find_package(FlagCV)` 链接迁移包的
消费者。Python 通过 `pytest tests/test_projective.py` 直接 JIT 执行 Triton
kernel，设置 `FLAGCV_PROJECTIVE_DATA` 和 `FLAGCV_PROJECTIVE_KERNELS` 可从迁移包
加载 provider 和 kernel。

BW1000（gfx936）、BW1101（gfx938）和 W7900D（gfx1100）已有统一集成版本的
公开 C++ 与安装后 Python JIT 实卡记录。HIP 的公开 C++ 验证入口为
`flagcv.hip.projective`，Python 使用同一测试文件。W7900D 的 C4 向量路径和
BW1000 的 C4 打包路径分别核对实际分派及相邻回退；不同版本的性能保留原身份。

共享 HIP 后端保留已有 Core 算子的公开接口和数值标准。S18 的九项 Imgproc
在三台 HIP 设备均有正规安装、公开 C++ 与安装 Python 的执行记录；
`e984188` 的整体安装证据与 `7fc934f`、`cad93ba` 的受影响滤波补验按
源码差异关联。BW1101 原 443 组中 C++ 验证 380 组、Python 验证 280 组，
尚缺 63／163 组，隔离性能未知。此处不把少量代表或验证平台写成完整性能验收。
按 [HIP 验收](hip-validation.md) 的平台范围维护支持状态。

## 后续批次

| 顺序 | 范围 | 状态 |
| --- | --- | --- |
| 1 | `bitwise_or`、`bitwise_xor`、`compare`、`min`、`max` | OR/XOR/min/max 有独立评测 PR 提案；公开 API 和验收待接入 |

AMD 新工具链与新设备继续按 HIP 交付流程验证。[AMD PR #3](https://github.com/flagos-ai/FlagCV/pull/3)

## 边框与阈值

| 算子 | C++ 语义范围 | 验证入口 |
| --- | --- | --- |
| `copyMakeBorder` | C1–C4 和已有 `Depth`。支持 CONSTANT、REPLICATE、REFLECT、REFLECT_101、WRAP。默认按孤立视图处理，显式父 ROI 可借用父图像素。 | `flagcv.{hip,nvidia}.border_threshold` 和 `tools/check_border_threshold.py` |
| `threshold` | 五种固定模式，8U/16U/16S/32F/64F。Otsu 支持 U8/U16 C1，Triangle 支持 U8 C1。自动模式提供设备结果与返回主机值的同步入口。 | 同上，含独立 OpenCV CPU 及阈值返回值检查 |

主题的设备和性能证据按实际平台与执行入口提交。FP32/FP64 参考采用固定 SSE3 OpenCV。

GaussianBlur 与 filter2D 的已接入范围和资源寿命见 [空间滤波](spatial-filters.md)。

## S18 记录范围

以下为已保存的构建和执行记录所覆盖的交付单元，不是穷举全部参数。
九算子 × 四平台 × 两入口共 72 单元。重建材料中三 HIP 的 54 单元有本地
原始入口记录；H20 的 18 单元保留回收前最终答复及历史报告观测，原最终
总索引尚未取回。原 449 行和去重 443 组分别统计，resizeBatch 的 48 项单列。

| 算子 | H20 | BW1000 gfx936 | BW1101 gfx938 | W7900D gfx1100 |
| --- | --- | --- | --- | --- |
| resize | C++／Python 记录 | C++／Python 记录 | C++／Python 记录 | C++／Python 记录 |
| warpAffine | 同上 | 同上 | 同上 | 同上 |
| cvtColor | 同上 | 同上 | 同上 | 同上 |
| warpPerspective | 同上 | 同上 | 同上 | 同上 |
| remap | 同上 | 同上 | 同上 | 同上 |
| copyMakeBorder | 同上 | 同上 | 同上 | 同上 |
| GaussianBlur | 同上 | 同上 | 同上 | 同上 |
| filter2D | 同上，含定向补验 | 同上，含定向补验 | 同上，含定向补验 | 同上，含定向补验 |
| threshold | 设备与同步返回分列 | 同上 | 同上 | 同上 |

数值差异按 [数值验收](numerical-tolerance.md) 的精确范围处理，不能推广成
所有整数的通用容差。原失败、重试、不可比基线及未测项继续保留。
