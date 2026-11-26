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

## Core 用例和计时

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

## 重放 Imgproc 已归档证据

使用既有 checker 的 `--imgproc-evidence RUNS_ROOT` 入口。该目录包含
原 H20 的 `flagcv-imgproc-stage1-perf-h20` 和
`flagcv-h20-stage2-native-baselines` 子目录。清单随包安装并校验这些
文件的摘要。将安装目录迁移后，可从任意目录运行下列命令。

```sh
python "$PREFIX/share/flagcv/benchmark/check_nvidia.py" \
  --imgproc-evidence "$RUNS_ROOT" --summary imgproc-archive-summary.json
```

输出保留九项的逐例 C++ 与 Python 记录及原始运行身份。可比速度比、
原生拒绝、语义差异、候选与 CPU 差异和缺项分别记录。自动阈值设备
结果与同步主机返回的计时单列。档案入口复核历史数据。
实时 NVIDIA 六项沿用原命令。合并检查可用同名选项附加归档复核。

归档默认仅生成报告，其中包含独立 CPU 状态、原生 CUDA 状态和作业身份。
通过 `--require-imgproc-scope` 可指定要核对的历史分组。
拒绝或语义差异可保留候选记录；必需 ID 缺失、身份不足或候选 CPU
失败会使该分组检查失败。全组为 `all`，可用 `geometry-large`
等更窄的分组。合并检查使用 `--imgproc-evidence` 时同时指定
`--imgproc-scope`。该检查只证明历史分组达到其验证条件。

## 从迁移安装运行 Imgproc

在 NVIDIA 构建时启用 `FLAGCV_BENCHMARK_IMGPROC` 和 `FLAGCV_BUILD_TESTS`，
并提供带 CUDA 算术、几何、颜色和滤波模块的 OpenCV CMake 包。
将该 FlagCV 安装迁移到 `PREFIX`。设置 Python 的 Triton 和同一 OpenCV
环境，在已取得物理设备锁的单张 SM90 卡上运行。

```sh
python "$PREFIX/share/flagcv/benchmark/check_nvidia.py" \
  --imgproc-live --prefix "$PREFIX" --case resize-0000 --case batch-00 \
  --python-benchmark-profile default --summary imgproc-live.json
```

省略 case 选项会运行九项代表及 Triangle 原生拒绝控制。可选 case 在
安装目录的 `benchmark/imgproc/cases.json` 中。resizeBatch 有独立的
48 条参数，FlagCV 每次调用处理整批，原生对照逐图提交。归档组
`resize-batch` 同样按真实批量用例计数。

C++ 使用安装的公开 API，Python 使用实际安装的 Triton kernel 和计划
描述。报告保留各自的原始计时和独立 CPU 检查。原生拒绝、输出布局变化
及语义不同均无正式比值。自动阈值的事件与同步主机返回时间分别记录。
此入口使用当前安装身份，完整冻结矩阵继续按原始运行身份归档。

## S18 Imgproc 收尾范围

九项 Imgproc 为 resize、warpAffine、cvtColor、warpPerspective、remap、
copyMakeBorder、GaussianBlur、filter2D、threshold。上文六 Core 的 112／92
用例口径保持不变，不能用它们代替九项 Imgproc 的数据。

H20 在 e984188 回放 13 组 BW 原冻结输入；7fc934f 的十项定向数据通过公开
C++、安装 JIT 与预先生成的整图参考；cad93ba 对既有 K13 四组进一步复验。
生产修复、独立参考、输入与目标 guard、安装源和分派分别有记录。
`--verify-only` 证明执行与正确性，不生成性能比值。

原 449 行与 443 个唯一参数组各留一份汇总；resizeBatch 的 48 项单列。
回收前报告的正式可比中位速度比（OpenCV CUDA / FlagCV）如下，缺失和不可比
不补零、不进入这些分母：

| 入口与指标 | 449 行口径可比数 | 443 组口径可比数 | 两口径中位速度比 |
| --- | ---: | ---: | --- |
| 公开 C++ Event | 179 | 177 | 0.684×／0.684× |
| Python kernel | 208 | 206 | 0.861×／0.859× |

这部分数值从回收前报告观测重建，原 H20 完整逐例文件尚未取回，不能用本表
重新验证原分布或生成缺失行。小于 1 表示该配对与指标下候选较慢。Graph、
设备事件、同步 host wall 的边界各自保留；不能以 kernel／Graph 结果冒充
公开调用或应用端到端性能。旧 C++ 0.689× 涉及 29 个来源不足的配对，保留
为历史诊断，不替代正式口径。

回收前原执行 Agent 报告的任务关闭 ID 为 apr-d57429a7078c。当前文档是恢复件，
原 Review JSON 和最终 72 单元总索引尚未取回，因此这里不是新的关闭凭据。
