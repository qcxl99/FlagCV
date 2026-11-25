# NVIDIA 验收

验收分为公开 C++ API、Python kernel、安装包和同设备性能四项。OpenCV CPU
作为参考，OpenCV CUDA 作为性能对照。环境要求包括 Linux/C++17、CMake 3.23+、
NVIDIA Toolkit 和 GPU、上游 Triton、PyTorch、pytest，以及含 `core` 和
`cudaarithm` 的 OpenCV 5 构建。Python 参考需可导入 `cv2`。环境和版本的记录
包含 OpenCV core/contrib 提交、GPU 型号和 PCI ID、驱动、工具链、Triton 版本、
样本数与本轮源码提交。

## 一次运行的验收入口

`python tools/check_merge.py --profile nvidia --opencv-dir "$OPENCV_CMAKE_DIR"
--output /path/to/new-nvidia-evidence` 运行一张卡的 NVIDIA 验收。用物理 GPU UUID
设置 `CUDA_VISIBLE_DEVICES`。在共享机器上优先由 KernelForge 的
`flagcv-acceptance` 入口检查空闲并锁卡。结果目录含每步命令与退出码、测试报告、
迁移后的安装包加载记录、原始性能数据和环境日志。另跑 host profile 和适用的
跨设备生命周期用例。下面保留逐步复跑方式，方便定位问题。

NVIDIA 合入默认使用 Python 100 ms 预热、100 ms 测量和两轮配对。
需要原有 1000 ms、三轮口径时，在合入命令加
`--python-benchmark-profile long`，该参数也可传给 KernelForge。
直接跑 pytest 时用 `--benchmark-profile long`。用检查器读取 long
报告时传 `--python-benchmark-profile long`。两个协议的基线分别保存。

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

合入入口生成 `performance-identity.json` 和 `performance-summary.json`。
检查器按 [冻结用例清单](../benchmark/nvidia_cases.json) 验收完整参数，核对
样本与中位数。主分支比较需要同协议的两个报告及身份文件，可加
`--identity`、`--previous-identity`、`--previous-cpp` 和 `--previous-python`。
设备、OpenCV CUDA 实际链接库、工具链和协议须匹配，两个候选的源码身份分别留存。
不匹配时报告原因，保留本次逐例报告。比值用于评审，固定阈值不阻断合入。

## 用例和计时

- C++ 正确性涵盖所有六个 API、已声明深度、1–4 通道、奇数及退化尺寸、各方向、
  独立字节 pitch、ROI、掩码、原地及部分重叠校验。检查输入、mask、padding、
  guard 和当前设备，验证缓存命中的调用在阻塞 stream 上立即返回。超出 CUDA
  grid Y 上限的高窄图像也由公开 API 测试。
- 注册失败测试给 NVIDIA 注册一个缺失的 kernel symbol。调用应报编译错误、
  保留目标字节，同一个 Handle 随后可执行未被平台注册的公版算子。
- C++ 性能覆盖 112 个共同 workload。使用 OpenCV perf 同用的
  `cv::getTickCount()`，对同一非默认流上的公开调用与等待计时。`host_single_us`
  每次调用单独等待，`host_batch_us` 在一批调用后等待并除以调用次数。
  预分配输入输出，先校验语义和边界，预热后交替测两实现。保留两项原始配对
  样本，汇总样本数、最小值、中位数、均值、总体标准差和几何均值。未执行
  OpenCV 的 `TEST_CYCLE` runner 或其异常值过滤。原有 Event 和提交时间列维持，
  各有独立时间边界。首次编译、分配、上传和下载均在计时区外。
- Python 92 个单通道 workload 使用 Triton `do_bench`，100 ms 测量
  并显式 `return_mode="median"`。默认 100 ms 预热、交替测两轮并取
  平均。long 用 1000 ms 预热、三轮中位数，对齐 FlagGems kernel 的
  预热与单轮统计。两种口径均以同卡 OpenCV CUDA 对照，JSON 保留
  配对样本与协议。Python 与 C++ 分别报告。
- 正确性另含 Python/C++ 共享整数向量、第二输入别名、全零与全满掩码、实际
  NVIDIA U8 tile 边界两侧的 packed 和 pitched 路径。不同线程各用一个 Handle
  和 stream 并验证释放隔离。Host CTest 用测试专用 backend 记录实际 Runtime 调用，
  验证编译、加载、缓存和释放顺序。安装包示例从迁移目录执行 add、bitwise_not 和 flip。
- 16U 乘法共同测速范围、`1×1 U16C1` 参考例外和确定性回归见 [数值验收](numerical-tolerance.md)。

## 已记录的 NVIDIA 基线

原协议下两个 H20 设备的工具链、逐例 CSV/JSON 和各算子最慢比值见
[2026-11-25 H20 基线](baselines/nvidia-h20-2026-11-25/README.md)。

仓内旧版归档用于复盘，缺少当前协议字段。较新的 1000 ms／三轮证据可在 long 模式下按原身份校验。与默认协议做主分支对比时，用默认协议重测目标基线。
