# 数值验收

OpenCV CPU 决定语义和精度。Python 参考使用 `reference/OP.py` 中的 `cv2`
实现，C++ 验收调用 `cv::Mat` 版本。六个已有 Python profile 的原 Torch 参考
用独立测试与 OpenCV CPU 交叉验证。OpenCV CUDA 用于同设备性能对照。

| 规则 | 当前用例 |
| --- | --- |
| 整数输出逐位一致 | 8U/16U add、subtract、multiply。新增整数和插值模式也从逐位一致开始。 |
| 存储位模式逐位一致 | bitwise_and、bitwise_not、flip，包含浮点存储的符号位与 NaN 载荷。 |
| 逐元素浮点误差 | 算术 32F 使用 `atol=0, rtol=1e-5`，16F 使用 `atol=0, rtol=1e-4`。要求满足 `abs(actual - expected) ≤ atol + rtol * abs(expected)`。同号无穷和同位置 NaN 单独比较。 |
| 可审查例外 | ±1 LSB 仅用于有可复现舍入差异的指定算子、参数和输入域。当前六个算子没有 LSB 例外。 |
| 结构化输出 | 由算子定义比较方法。提交输出结构、匹配规则与空输出、并列、顺序等边界用例。 |

Python 与 C++ 浮点阈值来自 `helpers/tolerance.json`。掩码外像素、ROI 外字节和
只读输入都执行独立的逐字节比较。

## 16U 乘法

本轮 x86 OpenCV 5.0.0 构建启用 CPU 优化分派时，`65535 × 65535` 可返回
0。关闭 `cv::setUseOptimized` 后结果按公开饱和语义为 65535。C++ 参考固定使用
该 CPU 路径并覆盖全 16U 输入域。OpenCV CUDA 的大乘积也观察到与这条 CPU
路径不同的结果，因而 16U 共同测速域将操作数限制为 `[0, 32767]`，结果依旧
逐位验证。性能输出以 `input_domain=u16_product_in_i32` 标记这些用例。
