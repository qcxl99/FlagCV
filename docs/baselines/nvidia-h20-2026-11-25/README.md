# NVIDIA H20 基线，2026-11-25

执行同设备 OpenCV CUDA 对照。C++ 每张卡 112 例、每例 11 对样本。Python 每张
卡 92 例、每例 3 对样本。C++ CSV 含流时间与提交时间中位数，Python JSON 含配对
样本和中位数。各行通过 OpenCV CPU 比较后才计时。

| 算子 | C++ 逐例最慢比值，卡 0 / 1 | Python 逐例最慢比值，卡 0 / 1 |
| --- | --- | --- |
| add | 1.000× / 1.013× | 1.349× / 1.345× |
| subtract | 0.926× / 0.927× | 1.884× / 1.843× |
| multiply | 0.996× / 1.013× | 1.849× / 1.808× |
| bitwise_and | 1.011× / 1.026× | 2.502× / 2.439× |
| bitwise_not | 0.949× / 0.978× | 2.654× / 2.691× |
| flip | 0.919× / 0.910× | 2.515× / 2.482× |

比值为 OpenCV CUDA 中位延迟除以 FlagCV 中位延迟。四份报告均达到逐例
`0.90×` 目标。[C++ 卡 0](cpp-0.csv) · [C++ 卡 1](cpp-1.csv) ·
[Python 卡 0](python-0.json) · [Python 卡 1](python-1.json)

## 环境

- Linux x86_64，GCC 13.3，CMake 3.28.3，CUDA Toolkit 12.8.93，驱动
  570.158.01。两张 NVIDIA H20 的 PCI 地址分别为 `C3:00.0` 和 `E3:00.0`。
- Python 3.12、PyTorch 2.10.0+cu128、Triton 3.6.0，NVIDIA 编译目标 `cuda:90:32`。
- OpenCV 5.0.0，core 提交 `40738fb16ceddb5fb3fea747585f7ce6abb0605b`。
  CUDA 对照由 contrib `2f5682baba1a8a7fd363247cf072c0ec7424d83e` 的
  `cudaarithm` 提供。构建的额外模块版本标记为 dirty，对应工作区的 Triton
  集成文件，`cudaarithm` 源码树与该提交一致。
- 安装包 smoke 的 Nsight Systems 记录了 1 次模块加载和 3 次 `flip_kernel`
  GPU launch。两个分配和两次传输来自树外示例。耗时样本独立于跟踪。
