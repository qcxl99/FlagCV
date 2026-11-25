# HIP 交付验收

`hip` 为 AMD ROCm 与 Hygon DTK 共用的 C++ Backend。已有调用方仍可用
`hygon` 名称编译和注册同一实现。公共 Handle、GpuMat、Stream、
lowering 和算子接口保持同一契约。插件对每次加载和提交使用 Handle
绑定的 HIP 设备，结束后恢复调用线程的设备，应用继续管理显存和流。

| 平台 | 当前证据 |
| --- | --- |
| W7900D gfx1100 / ROCm | 公共 C++ API、Python 正确性及同卡性能。 |
| BW1000 gfx936 / DTK | 公共 C++ API 和 Python 正确性。 |
| BW1101 gfx938 / DTK | 预览。构建和离线编译通过，待设备空闲后补实际 launch、正确性与生命周期。 |

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
Runtime、HIP 插件和编译数据来自该前缀。合入证据包含 C++ 六个算子的
OpenCV CPU 参考、数值例外、缓存、流与资源验证、拒绝用例以及
workload 和原始结果。HIP 编译阶段检查输出的 ABI，规则见
[Scratch ABI](architecture.md#scratch-abi-qualification)。仅完成离线编译的卡
仍需实际设备验收。

性能报告注明是否测 Python kernel 或公开 C++ API、计时和同步边界、
原生比较器的语义及分配边界。同卡有 ATen 等价实现时可配对。不可比的
路径分别标明原因。结果不能跨卡作为优化前后比值。
