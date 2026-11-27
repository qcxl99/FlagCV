# 性能测试

## 运行 Python kernel benchmark

从源码根目录运行。先选定对应后端、FlagTree 和一张可见设备，确保 Python
环境可导入 `cv2`。`--builtin` 使用生产 registry、provider、编译签名和 launch
plan；`--override` 注入的候选用于实验。候选先通过 OpenCV CPU 正确性检查，
输出在计时前预分配。

```sh
python -m pytest benchmark -m add --builtin --device cuda:0 \
  --baseline auto --record json --output /tmp/flagcv-add.json

python -m pytest benchmark -m add --builtin --device cuda:0 \
  --baseline auto --base-data /path/to/op_flagcv_perf_baseline.json \
  --record json --output /tmp/flagcv-add-with-baseline.json
```

用 `--list-cases` 列出用例，或用 `--case-id` 选择其中一个；旧 Python Case ID
仍可用。`--baseline none` 关闭原生比较，`--record json --output PATH` 保存结果。
`--benchmark-samples 1` 可快速筛选，`--benchmark-profile long` 可增加预热。
Core 用例来自 [workloads](../workloads/core.py)，Python benchmark 与 Optest recipe
共用其中的 Python 子集。

## 原生与外部比较

- **原生**：`--baseline auto` 使用符合语义的同卡 OpenCV CUDA 对照，或
  [已验证的 HIP ATen `out=` 用例](hip-validation.md#python-原生比较覆盖)。
  同卡加速比为 `原生中位延迟 / FlagCV 中位延迟`。
- **外部**：`--base-data` 读入 Optest 的 NVIDIA 实测基线，按 shape key 查找，
  核对完整配置、dtype 和计时协议。原始比值为 `NVIDIA 基线延迟 / FlagCV 延迟`；
  等效比值为 `原始比值 / 资源系数`，其中 `资源系数 = 目标实测资源值 / 实际 NVIDIA 参考设备实测资源值`。

正式评价优先使用通过的同卡原生比较。原生验证或计时失败单列，并保留原因；
外部 fallback 只供展示，不会改写正式失败。无合格换算时保留实测延迟、可比的
原始比值和缺失或不可比原因；汇总按来源与状态区分分母。

默认依据参考端 NVIDIA 的实测主导资源，只使用同资源、同精度且有依据的 DRAM
或标量 FP32 系数，不假定两端瓶颈相同。采集与 `do_bench` 使用一致的清缓存边界，
DRAM 需有实际流量（工作集可放入 L2 并不排除冷缓存下的 DRAM 访问）。明确受
launch 或缓存限制、合并采样多次调用、无法判断资源或缺必要计数器时不换算。可选
`--target-profile PATH` 提供双端诊断；目标用例存在但与参考不匹配时不换算，未提供
该用例诊断仍可按参考端计算。详细规则见[比较实现](../helpers/baseline.py)，采集见
[NVIDIA 指南](nvidia-validation.md#单算子原生基线采集)。`tools/check_merge.py`
也接受 `--base-data` 和 `--target-profile`。

## Python 计时协议

默认非 Graph Triton Event `do_bench`：25 ms 预热、100 ms 测量，取 median。
有合格原生对照时同卡交替测量三轮，保留逐轮耗时并各取中位数。`long` 使用
1000 ms 预热、100 ms 测量、三轮配对；`--warmup` 和 `--iter` 可改非 Graph
时间预算。原地路径使用 5 次预热和 25 个 Event 样本，在样本计时区外恢复输入。
报告记录实际使用的协议；不同输入域或不同协议的结果不能直接相比。

## C++ API benchmark

按对应[后端指南](testing.md#设备测试)配置 SDK 与 OpenCV 5 core，在 CMake
配置时加 `-DFLAGCV_BUILD_BENCHMARKS=ON` 并构建。以 NVIDIA 为例（AMD/Hygon
使用对应目录和目标名）：

```sh
./build-nvidia/backends/nvidia/validation/flagcv_benchmark_nvidia 0 /tmp/flagcv-cpp.csv
./build-nvidia/backends/nvidia/validation/flagcv_benchmark_nvidia 0 /tmp/flagcv-cpp-diag.csv 21 --diagnostics
```

每例先完成编译、预热和 OpenCV CPU 正确性检查，再用预分配缓冲区在同一 stream
采样。Device Latency 用设备 Event 覆盖单次执行；API Latency 从 Host 公开调用前计至
该 stream 上该次工作完成，采样前排空 stream。`--diagnostics` 另采 Host submit
和 batch；`add`、`bitwise_not`、`flip` 的代表性小用例也对照显式 Handle 和便利接口。

C++ 报告绝对延迟、同卡历史对照或已验证等价的同卡原生对照。前后对比使用同卡、
同用例、同输入域和同协议。Python 外部基线的资源系数不用于 C++。原生 0.90×、
普通等效 80% 和关键算子等效 95% 只是参考；验收以正确性与报告完整性为门槛。

[原协议与比较细节归档](https://github.com/qcxl99/FlagCV/blob/adc01fda1014a530f2a844ad28746829e46a2da0/docs/benchmark.md)
保留该修订的内部字段与用例统计。
