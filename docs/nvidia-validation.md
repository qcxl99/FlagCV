# NVIDIA 验收

本轮编译器固定为 `flagtree-nvidia==0.7.1.dev20261125`，对应的模块报告
Triton 3.6.0。Optest H800 配置使用
`harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-dev`。用于采集时，
环境还须提供面向实际 GPU 构建的 OpenCV 5 core、CUDA `cudaarithm` 和
Nsight Compute。记录实际运行的镜像 digest、GPU、CUDA SDK、驱动以及
Python 和编译子进程的发行版与模块路径。以下 wheel 来自 [FlagOS
软件索引](https://resource.flagos.net/repository/flagos-pypi-daily/simple/flagtree-nvidia/)。
Optest 镜像和软件版本见 [H800 配置](https://github.com/flagos-ai/optest/blob/c0f9bdc84449a5722d09e58df255499e08f365e9/.github/configs/backends/H800.yaml)。

```sh
python -m pip install --no-deps --target /opt/flagcv-flagtree \
  --index-url https://resource.flagos.net/repository/flagos-pypi-daily/simple \
  flagtree-nvidia==0.7.1.dev20261125
export PYTHONPATH="/opt/flagcv-flagtree:${PYTHONPATH:-}"
export FLAGCV_PYTHON="$(command -v python)"
"$FLAGCV_PYTHON" -c 'import importlib.metadata as m, triton; print(m.version("flagtree-nvidia"), triton.__version__, triton.__file__)'
```

## 验收和原生比较

在 CUDA SDK 及兼容的 OpenCV 构建环境中，指定一张空闲 GPU 和
`OpenCVConfig.cmake` 所在目录，运行 C++ 和 Python 测试、迁移安装和
Core benchmark。

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" python tools/check_merge.py --profile nvidia \
  --opencv-dir "$OpenCV_DIR" --output /tmp/flagcv-nvidia
```

手工运行 Optest 采集时先构建六算子 OpenCV CUDA bridge。该适配器
使用同一原生 stream 和调用方预分配的输出。CPU 正确性由 `cv2` 提供。

```sh
cmake -S adapters/opencv_cuda -B build-opencv-cuda -DOpenCV_DIR="$OpenCV_DIR"
cmake --build build-opencv-cuda --parallel
export FLAGCV_OPENCV_LIBRARY="$PWD/build-opencv-cuda/libflagcv_opencv_cuda.so"
python /path/to/optest/src/nv_baseline/collect_baseline_nvidia.py \
  --device 0 --ops-dir "$PWD/baseline" --ops add --check
python -m pytest benchmark -m add --builtin --device cuda:0 \
  --case-id 'benchmark/test_add_benchmark.py::test_add_benchmark::core::float32::0' \
  --record json --output /tmp/flagcv-add-check.json
python /path/to/optest/src/nv_baseline/collect_baseline_nvidia.py \
  --device 0 --ops-dir "$PWD/baseline" --ops add --mode full \
  --reference-chip "$NVIDIA_BASE_CHIP" --output /tmp/flagcv-add-baseline.json
```

先校验单个 add 绑定的桥接、CPU 正确性、真实采集及汇总，再扩展到全量
Core。将采集 JSON 放入 Optest 的 `data/`，并在 Optest 仓库配置 FlagCV
的运行环境和采集路径。计时、基线匹配与系数规则见 [性能测试](benchmark.md)。

当前整改版本已完成 SM90 离线编译，六算子及掩码入口的 PTX 签名和
scratch ABI 检查通过。NVIDIA 实卡 launch、迁移安装、原生采集及新协议
的性能对照待复验。[2026-11-25 H20 历史证据](baselines/nvidia-h20-2026-11-25/README.md)
记录原 Triton 工具链和原计时协议。
