# NVIDIA 验收

验收分为公开 C++ API、Python kernel、安装包和同设备性能四项。OpenCV CPU
作为参考，OpenCV CUDA 作为性能对照。环境要求包括 Linux/C++17、CMake 3.23+、
NVIDIA Toolkit 和 GPU、上游 Triton、PyTorch、pytest，以及含 `core` 和
`cudaarithm` 的 OpenCV 5 构建。Python 参考需可导入 `cv2`。环境和版本的记录
包含 OpenCV core/contrib 提交、GPU 型号和 PCI ID、驱动、工具链、Triton 版本、
样本数与本轮源码提交。

## 复跑

在 FlagCV 根目录执行。替换 `OPENCV_CMAKE_DIR`，使它指向含
`OpenCVConfig.cmake` 的目录。选择一个或两个空闲 GPU。只有一张卡时将
`FLAGCV_TEST_NVIDIA_DEVICES` 改为 `1`，并执行设备 0 的命令。

```sh
export OPENCV_CMAKE_DIR=/path/to/opencv/lib/cmake/opencv5
export FLAGCV_PYTHON="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0,1

cmake -S . -B build-nvidia -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_WARNINGS_AS_ERRORS=ON -DFLAGCV_TEST_NVIDIA=ON \
  -DFLAGCV_TEST_NVIDIA_DEVICES=2 -DFLAGCV_BENCHMARK_NVIDIA=ON \
  -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-nvidia --parallel
ctest --test-dir build-nvidia --output-on-failure
python -m pytest -q tests/core
python -m pytest -q tests/test_*.py --builtin --device cuda:0
python -m pytest -q tests/test_*.py --builtin --device cuda:1

cmake --install build-nvidia --prefix "$PWD/build-nvidia/install"
cmake -S tests/consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build-nvidia/install"
cmake --build build-consumer
build-consumer/flagcv_install_consumer

cmake -S adapters/opencv_cuda -B build-opencv-cuda \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-opencv-cuda --parallel
export FLAGCV_OPENCV_LIBRARY="$PWD/build-opencv-cuda/libflagcv_opencv_cuda.so"
mkdir -p build-nvidia/results
for device in 0 1; do
  build-nvidia/tests/flagcv_benchmark_nvidia "$device" \
    "build-nvidia/results/cpp-$device.csv" 11
  python -m pytest -q benchmark --builtin --device "cuda:$device" \
    --record json --output "build-nvidia/results/python-$device.json"
  python benchmark/check_nvidia.py --cpp "build-nvidia/results/cpp-$device.csv" \
    --python "build-nvidia/results/python-$device.json"
done
```

已交付路径的回归比较可向 checker 再传 `--previous-cpp` 和
`--previous-python`，使用同机、同设备、同模式的主分支报告。冻结新基线时按
[合入标准](../CONTRIBUTING.md) 逐例审查。

## 用例和计时

- C++ 正确性涵盖所有六个 API、已声明深度、1–4 通道、奇数及退化尺寸、各方向、
  独立字节 pitch、ROI、掩码、原地及部分重叠校验。检查输入、mask、padding、
  guard 和当前设备，验证缓存命中的调用在阻塞 stream 上立即返回。超出 CUDA
  grid Y 上限的高窄图像也由公开 API 测试。
- 注册失败测试给 NVIDIA 注册一个缺失的 kernel symbol。调用应报编译错误、
  保留目标字节，同一个 Handle 随后可执行未被平台注册的公版算子。
- C++ 逐例性能为 112 个共同 workload，包括 64×64、61×67、VGA、1080p、4K
  和多通道 ROI。每例比较同一非默认流上的预分配缓冲区，成功通过 OpenCV CPU
  校验、输入与边界检查后再计时。交替测量两实现，报告每次 batch 的 CUDA event
  平均耗时的样本中位数，以及调用提交的主机时间中位数。流时间包含提交造成的空隙，
  适用于 API 端到端评价。首次编译、分配、上传和下载在计时区外。
- Python 性能包含 92 个单通道 workload。用 FlagGems 兼容 JSON 字段保存 paired
  样本和中位数。使用 Triton event 计时，数据在测试前上传。Python 和 C++ 的
  计时路径与所涵盖布局分别记录。
- 16U 乘法共同测速范围以及全域 CPU 验收见 [数值验收](numerical-tolerance.md)。

## 已记录的 NVIDIA 基线

本轮两个 H20 设备的工具链、逐例 CSV/JSON 和各算子最慢比值见
[2026-11-25 H20 基线](baselines/nvidia-h20-2026-11-25/README.md)。
