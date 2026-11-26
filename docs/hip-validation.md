# HIP 交付验收

`amd` 对应 ROCm 的 gfx1100 wave32，`hygon` 对应 DTK 的 gfx936/gfx938 wave64。
公开选择使用厂商名。旧的 `Handle("hip")` 与 `FLAGCV_BACKENDS=hip` 会报错，
已有调用方需显式迁移。Triton target 仍采用 `hip:架构:波宽`。插件保留原
Handle、GpuMat、Stream 和调用线程设备恢复契约。

六算子拆分资格基于 `e48810e`。各平台使用单张可见设备和对应 SDK。

| 平台 | 六算子验证 |
| --- | --- |
| AMD W7900D gfx1100 / ROCm 7.2.4 | 公开 C++ 矩阵、Python 128/128、迁移包树外消费。 |
| Hygon BW1000 gfx936 / DTK 26.04 | 公开 C++ 矩阵和安装 Python 122/122 从拆分版本 00599d9 继承；e48810e 迁移消费实测通过。 |
| Hygon BW1101 gfx938 / DTK 26.10 | 公开 C++ 184/184、安装 Python 128/128、迁移包消费与流验证。 |

BW1000 的继承依据为两个版本的 GPU 分派、算子源码与全部已登记 target
计划相同。该表限定本次验证的设备与工具链。没有新增多卡或性能测量。

## 复跑

选空闲卡并按物理设备取得锁，记录 PCI、GPU 身份、SDK、FlagTree 和
OpenCV CPU 构建。每个候选独立使用编译输出与缓存。通过 `hip_DIR` 选择对应 SDK 的 CMake 配置，ROCm 和 DTK 分别使用
实际 SDK 的 `hip::host`。将 `OPENCV_CMAKE_DIR` 指向 OpenCV 5 core
的构建或安装目录。Python 测试需要同构建可导入的 `cv2`。

```sh
# Hygon DTK SDK
cmake -S . -B build-hygon -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=hygon -DFLAGCV_TEST_HYGON=ON \
  -Dhip_DIR="$DTK_HIP_CMAKE_DIR" -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-hygon --parallel
ctest --test-dir build-hygon -L host --output-on-failure
ctest --test-dir build-hygon -L hygon --output-on-failure
GEMS_VENDOR=hygon python -m pytest -q tests/test_*.py --builtin --device cuda:0

# AMD ROCm SDK
cmake -S . -B build-amd -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=amd -DFLAGCV_TEST_AMD=ON \
  -Dhip_DIR="$ROCM_HIP_CMAKE_DIR" -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-amd --parallel
ctest --test-dir build-amd -L host --output-on-failure
ctest --test-dir build-amd -L amd --output-on-failure
GEMS_VENDOR=amd python -m pytest -q tests/test_*.py --builtin --device cuda:0
```

分别安装到独立前缀，再迁移安装目录。用 `tests/consumer` 编译树外示例。
Hygon 设置 `FLAGCV_CONSUMER_HYGON=ON`，AMD 设置 `FLAGCV_CONSUMER_AMD=ON`。
将 `CMAKE_PREFIX_PATH` 指向对应的迁移前缀，并选择相同 SDK 的 `hip_DIR`。
运行树外示例与公开 API 矩阵，核对实际加载的
Runtime、HIP 插件和编译数据来自该前缀。合入证据包含 C++ 六个算子的
OpenCV CPU 参考、数值例外、缓存、流与资源验证、拒绝用例以及
workload 和原始结果。HIP 编译阶段检查输出的 ABI，规则见
[Scratch ABI](architecture.md#scratch-abi-qualification)。仅完成离线编译的卡
仍需实际设备验收。

性能报告注明是否测 Python kernel 或公开 C++ API、计时和同步边界、
原生比较器的语义及分配边界。同卡有 ATen 等价实现时可配对。不可比的
路径分别标明原因。结果不能跨卡作为优化前后比值。
