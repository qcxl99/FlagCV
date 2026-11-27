# NVIDIA Backend Guide

## 工具链

[NVIDIA provider](../backends/nvidia/compiler.py)要求
`flagtree-nvidia==0.7.1.dev20261125` 及同一发行包提供的 Triton 3.6.0。
从 [FlagOS 软件索引](https://resource.flagos.net/repository/flagos-pypi-daily/simple/flagtree-nvidia/)
安装到独立目录，并核对编译子进程使用的发行版和模块路径：

```sh
python -m pip install --no-deps --target /opt/flagcv-flagtree \
  --index-url https://resource.flagos.net/repository/flagos-pypi-daily/simple \
  flagtree-nvidia==0.7.1.dev20261125
export PYTHONPATH="/opt/flagcv-flagtree:${PYTHONPATH:-}"
export FLAGCV_PYTHON="$(command -v python)"
"$FLAGCV_PYTHON" -c 'import importlib.metadata as m, triton; print(m.version("flagtree-nvidia"), triton.__version__, triton.__file__)'
```

确认输出版本与上述 pin 一致，且 `triton.__file__` 指向这份 wheel
安装的 `triton/__init__.py`，而不是另一个 Triton 安装。

## 构建和测试

使用 CUDA Toolkit 和兼容驱动。C++ GPU 功能测试需要 OpenCV 5 `core`；
原生 OpenCV CUDA 对照与 bridge 还需要面向目标 GPU 构建的 `cudaarithm`。
Python 验证需要 CUDA Torch 和可导入的 CPU `cv2`。将 `OpenCV_DIR` 设为
`OpenCVConfig.cmake` 所在目录，并选择一张可用的卡（`GPU_UUID`）。

```sh
cmake -S . -B build-nvidia -DCMAKE_BUILD_TYPE=Release \
  -DFLAGCV_BACKENDS=nvidia -DFLAGCV_BUILD_TESTS=ON -DFLAGCV_TEST_NVIDIA=ON \
  -DOpenCV_DIR="$OpenCV_DIR"
cmake --build build-nvidia --parallel
ctest --test-dir build-nvidia -L host --output-on-failure
CUDA_VISIBLE_DEVICES="$GPU_UUID" ctest --test-dir build-nvidia -L nvidia --output-on-failure
CUDA_VISIBLE_DEVICES="$GPU_UUID" GEMS_VENDOR=nvidia \
  python -m pytest -q tests/test_*.py --builtin --device cuda:0
```

从构建到迁移安装和性能的验证入口（另需 `cudaarithm`）见[测试指南](testing.md)：

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" python tools/check_merge.py --profile nvidia \
  --opencv-dir "$OpenCV_DIR" --output /tmp/flagcv-nvidia
```

## 单算子原生基线采集

从 FlagCV 源码根目录执行。Bridge 复用调用方的预分配输出和原生 stream。
使用含 [portable collector 注册](https://github.com/qcxl99/optest/commit/baf494fdc70d9a25506c6717a3cfcc4c0666564d)
的 Optest checkout；实际检查和采集还需 `ncu` 在 `PATH` 上。

```sh
cmake -S adapters/opencv_cuda -B build-opencv-cuda -DOpenCV_DIR="$OpenCV_DIR"
cmake --build build-opencv-cuda --parallel
export FLAGCV_OPENCV_LIBRARY="$PWD/build-opencv-cuda/libflagcv_opencv_cuda.so"
COLLECT=/path/to/optest/src/nv_baseline/collect_registered.py
python "$COLLECT" FlagCV "$PWD" --ops add --plan
CUDA_VISIBLE_DEVICES="$GPU_UUID" python "$COLLECT" FlagCV "$PWD" --ops add --check
CUDA_VISIBLE_DEVICES="$GPU_UUID" python "$COLLECT" FlagCV "$PWD" --ops add \
  --output /tmp/flagcv-add-baseline.json --report-dir /tmp/flagcv-add-ncu
```

`--plan` 只解析注册和命令，无需 GPU；`--check` 做环境预检与 recipe smoke，不是
性能采集。默认 `H800` 来自 Optest 注册；换用 H20 时在相应命令中显式添加
`--reference-chip H20` 并选择 H20。此参数指定参考设备身份，不选择物理卡，
预检会核对实际设备。缺少合格参考系数时不做等效换算。候选侧复跑及 JSON
用法见[性能测试](benchmark.md)。

## Scratch ABI

[产物校验](../compiler/flagcv_codegen/scratch_abi.py)要求 global 和 profile
scratch 大小均显式为零，并比对 PTX 入口的参数布局，包括两个尾部 64 位指针；
Runtime 在加载前核对尾参数数量。见[通用合同](architecture.md#scratch-abi-qualification)。

设备验证与采集状态见[平台支持清单](operators.md#平台支持范围)。
[2026-11-25 H20 历史基线](baselines/nvidia-h20-2026-11-25/README.md)保留原工具链和计时协议。
