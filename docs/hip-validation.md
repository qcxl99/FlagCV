# AMD/Hygon Backend Guide

公开后端名为 `amd`（ROCm）和 `hygon`（DTK）；`Handle("hip")` 和
`FLAGCV_BACKENDS=hip` 不再接受。两端分别使用对应 SDK 的 `hip::host`、
独立构建目录和匹配的 FlagTree。Triton target 仍使用 `hip:架构:波宽`。
已验证的设备、工具链及功能范围见[平台支持清单](operators.md#平台支持范围)。

## 构建和测试

在设备测试前选择对应 SDK 和 FlagTree，隔离可见设备、构建输出和缓存。
`OPENCV_CMAKE_DIR` 指向 OpenCV 5 core 的 CMake 配置目录；Python 功能验证
另需可导入的 CPU `cv2` 和对应 SDK 的 PyTorch 环境。

```sh
cmake -S . -B build-amd -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=amd -DFLAGCV_BUILD_TESTS=ON -DFLAGCV_TEST_AMD=ON \
  -Dhip_DIR="$ROCM_HIP_CMAKE_DIR" -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-amd --parallel
ctest --test-dir build-amd -L host --output-on-failure
ctest --test-dir build-amd -L amd --output-on-failure
GEMS_VENDOR=amd python -m pytest -q tests/test_*.py --builtin --device cuda:0

cmake -S . -B build-hygon -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=hygon -DFLAGCV_BUILD_TESTS=ON -DFLAGCV_TEST_HYGON=ON \
  -Dhip_DIR="$DTK_HIP_CMAKE_DIR" -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-hygon --parallel
ctest --test-dir build-hygon -L host --output-on-failure
ctest --test-dir build-hygon -L hygon --output-on-failure
GEMS_VENDOR=hygon python -m pytest -q tests/test_*.py --builtin --device cuda:0
```

## 安装与集成测试

安装到独立目录，将安装目录移到新位置，再用 `tests/consumer` 中的示例验证第三方项目集成。
以下命令以 AMD 为例。

```sh
cmake --install build-amd --prefix "$AMD_STAGE"
mv "$AMD_STAGE" "$AMD_PREFIX"
cmake -S tests/consumer -B build-consumer-amd -DCMAKE_PREFIX_PATH="$AMD_PREFIX" \
  -DFLAGCV_CONSUMER_NVIDIA=OFF -DFLAGCV_CONSUMER_AMD=ON \
  -Dhip_DIR="$ROCM_HIP_CMAKE_DIR"
cmake --build build-consumer-amd --parallel
./build-consumer-amd/flagcv_install_consumer
```

以上目录应为新的绝对路径。Hygon 用独立的构建/安装目录、
`FLAGCV_CONSUMER_HYGON=ON` 和 DTK 的 `hip_DIR`，不同时启用 AMD。
核对加载的 Runtime、插件和编译数据均来自新的安装目录。需要同时检查公开 C++ API、
安装包和性能时，使用 `tools/check_merge.py --profile amd` 或 `--profile hygon`，
传入对应的 `--hip-dir`、`--opencv-dir` 和新的 `--output` 目录。

## Python 原生比较覆盖

`--baseline auto` 的 `aten_hip_out` 仅在核实为 Hygon BW1000（`BW`、`gfx936`、
80 CU）的设备上启用。合格域是同卡、同形状、紧凑单通道、无掩码的 `uint8`
AND/NOT 和 `float32` add/subtract/单位缩放 multiply。操作数须有独立存储、零
storage offset 和 16 字节对齐的基地址。逐例与 OpenCV CPU 核对；候选须在当前
stream 上写入预分配输出。外部候选可复用合格的 prepared launcher；未经验证的
callable 边界或域外的 `out=` 用例仍为 `pending`。见 [ATen adapter](../adapters/aten_hip.py)。
运行验证或计时失败为 `failed`。`uint8` 算术回绕而非饱和，`torch.flip` 会分配
输出，均不等价。这是 Python 比较资格，不是 C++ API 资格。

W7900D 和 BW1101 的 ATen 比较器未启用。没有合格原生比较时仍记录实测延迟；
用 `--base-data PATH` 提供 Optest 外部基线，比较规则见[性能测试](benchmark.md)。
资源系数由 Optest 按实测维护，不将设备理论规格填入系数数据库。

## Scratch ABI

AMD 与 Hygon 使用独立且有界的 Triton 3.6.0 compiler/target 集合。
[HIP 产物校验](../compiler/flagcv_codegen/scratch_abi.py)检查 LLVM 入口签名、
两个尾部指针的 `readnone` 和显式为零的 profile scratch。部分 HIP 编译器省略
global scratch 大小，此时仍须逐产物证明两指针未使用；非零 global scratch 被拒绝。
通用合同见 [Scratch ABI qualification](architecture.md#scratch-abi-qualification)。

## 历史记录

[固定版本的 HIP 记录](https://github.com/qcxl99/FlagCV/blob/adc01fda1014a530f2a844ad28746829e46a2da0/docs/hip-validation.md)
保留后端拆分与迁移证据、ATen 实卡 trace、`fd6153c` 验收结果及 BW1101 规格表；
这些记录绑定原候选和协议，不代表其他组合已验证。
