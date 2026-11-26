# HIP 交付验收

`amd` 对应 ROCm 的 gfx1100 wave32，`hygon` 对应 DTK 的 gfx936/gfx938 wave64。
公开选择使用厂商名。旧的 `Handle("hip")` 与 `FLAGCV_BACKENDS=hip` 会报错，
已有调用方需显式迁移。Triton target 仍采用 `hip:架构:波宽`。插件保留原
Handle、GpuMat、Stream 和调用线程设备恢复契约。

## 拆分版历史记录

以下六算子拆分证据基于 `e48810e`。各平台使用单张可见设备和对应 SDK。

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

## 六算子整改验证

以下资格绑定整改候选 `fd6153c` 及各行设备和工具链。三端均实测双卡公开 API、
原生 Stream、调用方设备状态恢复和迁移安装后的树外调用。

| 平台 | 工具链 | 功能与安装 |
| --- | --- | --- |
| Hygon BW1000 `gfx936` | DTK 26.04、FlagTree `0.7.0+hcu3.6` | 六算子双卡 C++、Python 128/128、树外消费通过。 |
| Hygon BW1101 `gfx938` | DTK 26.10、FlagTree `0.7.1a1+hcu3.6` | C++ 每卡 184/184、Python 128/128、树外消费通过。 |
| AMD W7900D `gfx1100` | ROCm 7.2.4、FlagTree `0.7.0+amd3.6` | C++ 每卡 184/184、Python 128/128、树外消费通过。 |

BW1000 复跑了同卡、同协议的整改前后 112 个 C++ Core 用例，并核对 92 个
Python Core 用例与 C++ 分派。BW1101 和 W7900D 分别保存 112 个 C++ Core、
3 组便利接口配对及 92 个 Python Core 的实测延迟。BW1101 的延迟属于共享卡
观测。两端缺少可用的原生或等效性能基线，相关速度比留空。三端功能资格与外部
性能基线状态分别记录。

```sh
HIP_VISIBLE_DEVICES=0,1 python tools/check_merge.py --profile hygon --devices 2 \
  --hip-dir "$DTK_HIP_CMAKE_DIR" --opencv-dir "$OPENCV_CMAKE_DIR" \
  --output /tmp/flagcv-bw1000
```

BW1101 规格参考为表内类型 8，`NMZ_BW1101_CU64_144G`，64 CU、
144G、基础频率 1.20 GHz、标注 boost 1.35 GHz。

| 规格表项目 | Base / Peak |
| --- | ---: |
| Matrix（TFLOPS） | 315 / 354 |
| i8gemm（表头 TFLOPS） | 629 / 708 |
| fp8gemm（表头 TFLOPS） | 629 / 708 |
| Vector hgemm | 20 / 21 |
| Vector i8gemm | 39 / 43 |

向量列未单列单位。实际瓶颈、精度和测量口径匹配时才使用测量系数。
三个 HIP 平台缺少合格换算系数时均保留实测延迟及可比的原始延迟比。
