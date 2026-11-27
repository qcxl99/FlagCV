# 数值验收

OpenCV CPU 决定语义和精度。Python 参考使用 `reference/<op>.py` 中的 `cv2`
实现，C++ 验收调用 `cv::Mat` 版本。OpenCV CUDA 用于同设备性能对照。

| 规则 | 当前用例 |
| --- | --- |
| 整数输出逐位一致 | 8U/16U add、subtract、multiply。新增整数和插值模式也从逐位一致开始。 |
| 存储位模式逐位一致 | bitwise_and、bitwise_not、flip，包含浮点存储的符号位与 NaN 载荷。 |
| 逐元素浮点误差 | 算术 32F 使用 `atol=0, rtol=1e-5`，16F 使用 `atol=0, rtol=1e-4`。要求满足 `abs(actual - expected) ≤ atol + rtol * abs(expected)`。同号无穷和同位置 NaN 单独比较。 |
| 可审查例外 | 只有明确声明算子、参数、输入域及可复现舍入差异时才允许 ±1 LSB。当前未声明 LSB 例外。 |
| 结构化输出 | 由算子定义比较方法。提交输出结构、匹配规则与空输出、并列、顺序等边界用例。 |

Python 与 C++ 浮点阈值来自 [tolerance.json](../helpers/tolerance.json)。掩码外像素、ROI 外字节和
只读输入都执行独立的逐字节比较。

## 16U 乘法

C++ 参考关闭 OpenCV CPU 优化分派。在固定的 x86 OpenCV 5.0.0 构建上，
关闭优化后，两个操作数都为 65535 时，`1×1 U16C1` 仍返回 0，`1×8 U16C1`
返回 65535。启用优化时，本次对照的两个尺寸都返回 0。

**正式参考例外只适用于单位缩放的 `1×1 U16C1` 逐元素乘法。** 先将两个
操作数转为无符号 32 位整数，再取 `min(a * b, 65535)`，要求输出逐位一致。
其余已验证尺寸仍与关闭优化的 OpenCV CPU 比较。这是参考实现例外，不是容差放宽。

OpenCV CUDA 的大乘积也观察到差异，因此同设备性能对照仍将两个操作数
限定在 `[0, 32767]`，并在计时前验证正确性。性能输出用
`input_domain=u16_product_in_i32` 标记这一范围。
