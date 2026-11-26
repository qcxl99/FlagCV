# HIP 交付验收

`hip` 为 AMD ROCm 与 Hygon DTK 共用的 C++ Backend。已有调用方仍可用
`hygon` 名称编译和注册同一实现。公共 Handle、GpuMat、Stream、
lowering 和算子接口保持同一契约。插件对每次加载和提交使用 Handle
绑定的 HIP 设备，结束后恢复调用线程的设备，应用继续管理显存和流。

| 平台 | 当前证据 |
| --- | --- |
| W7900D gfx1100 / ROCm | 公共 C++ API、Python 正确性及同卡性能。 |
| BW1000 gfx936 / DTK | 正规迁移安装、九项 Imgproc 双入口、流／ABI 和受影响数值路径；性能沿原版本与可比范围记录。 |
| BW1101 gfx938 / DTK | 已有实际 launch、正规安装、迁移消费者、九项双入口和数值补验。S16 主责验证，隔离性能未知；历史矩阵仍有未测组。 |

## 复跑

选空闲卡并按物理设备取得锁，记录 PCI、GPU 身份、SDK、FlagTree 和
OpenCV CPU 构建。每个候选独立使用编译输出与缓存。将
`HIP_CMAKE_DIR` 指向当前 SDK 的 CMake 配置，ROCm 和 DTK 环境均使用
实际 SDK 的 `hip::host`。将 `OPENCV_CMAKE_DIR` 指向 OpenCV 5 core
的构建或安装目录。Python 测试需要同构建可导入的 `cv2`。

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=hip -DFLAGCV_TEST_HIP=ON \
  -DFLAGCV_WARNINGS_AS_ERRORS=ON -Dhip_DIR="$HIP_CMAKE_DIR" \
  -DOpenCV_DIR="$OPENCV_CMAKE_DIR"
cmake --build build-hip --parallel
ctest --test-dir build-hip -L host --output-on-failure
ctest --test-dir build-hip -L hip --output-on-failure
python -m pytest -q tests/test_*.py --builtin --device cuda:0
```

安装后迁移安装前缀。用 `tests/consumer` 编译树外示例（开启
`-DFLAGCV_CONSUMER_HIP=ON`，将 `CMAKE_PREFIX_PATH` 指向迁移后前缀）。
运行树外示例与公开 API 矩阵，核对实际加载的
Runtime、HIP 插件和编译数据来自该前缀。合入证据保留已有 Core 六项，并分别登记九项 Imgproc 的公开 C++ 与安装 Python、
OpenCV CPU 参考、数值例外、缓存、流与资源验证、拒绝用例以及
workload 和原始结果。HIP 编译阶段检查输出的 ABI，规则见
[Scratch ABI](architecture.md#scratch-abi-qualification)。仅完成离线编译的卡
仍需实际设备验收。

性能报告注明是否测 Python kernel 或公开 C++ API、计时和同步边界、
原生比较器的语义及分配边界。同卡有 ATen 等价实现时可配对。不可比的
路径分别标明原因。结果不能跨卡作为优化前后比值。

## S18 已保存的集成验证

共享源以 e984188 完成正规安装资格，7fc934f 修复受证书约束的 U8 相消路径，
cad93ba 增加 K13 机制参考工具和文档。各阶段的实际源码、库、对象与设备身份
保留，不将旧性能重新标为新版本。每个 BW 安装的两入口分别绑定本机同一套 Galaxy HIP，BW1000 为 DTK 26.04、
BW1101 为 DTK 26.10；W7900D 使用 ROCm。正规工具核验各自的实际运行库，
早期临时诊断副本不作为正式通过依据。

| 平台 | 正规联合入口（e984188） | 最新受影响项 |
| --- | ---: | --- |
| BW1000 | 17 请求／16 独立案例，双入口 | 7fc 原包十例与保留控制；cad 四组 K13，两入口全图符合声明参考 |
| BW1101 | 13 独立案例，双入口 | 7fc 十例与真半步；cad 四组 K13，两入口通过 |
| W7900D | 12 行，双入口 | 7fc 十例与五项原控制；cad 四组 K13，两入口通过 |

三台均有 host 6/6、迁移安装与树外消费者证据。表中数字不是 449／443 的
全量覆盖。BW1101 历史参数仍为 C++ 380/443、Python 280/443；未测 63/163
明确保留。共享占用下有界功能验证不能转成隔离性能结果。

安装的 checker 支持以下无计时验证形式，仍须由调用方先核卡与取得物理锁：
`GPU_UUID` 为已取得锁的物理卡的 HIP runtime UUID，不能用逻辑卡号替代。

```sh
python "$PREFIX/share/flagcv/benchmark/check_nvidia.py" \
  --imgproc-live --backend hip --device-uuid "$GPU_UUID" --prefix "$PREFIX" --verify-only \
  --case filter-37-19-2-11-3-0 --summary hip-filter-verify.json
```

U16 三尺度原例的 703 个输出为 1；S17 有限正饱和保留 65535，固定 CPU0
单列诊断。真半步、微小正尾和 K13 依据各自独立参考验收，规则见
[数值规范](numerical-tolerance.md)。这些新补验没有生成新的发布性能采样。
