# 性能测试

`workloads/` 定义 Core 用例。C++ 构建生成 112 个 Core 用例，内部
Python 和六个 Optest recipe 共用 92 个 Core 用例。Case ID 由规范化配置
生成，包含算子、variant、dtype、shape、行距、布局、算子参数、别名、输出
分配及输入域。计时协议独立记录。旧的 Python Case ID 可以继续用于
`--case-id`，用例清单同时提供新旧 ID。

## Python kernel

`--builtin` 复用 C++ registry、provider、编译签名和 launch plan。
逐例记录编译器身份、产物哈希、variant、编译常量、warp 数和 grid。
外部候选的报告标为实验数据。计时前用 OpenCV CPU 验证结果并预分配输出。

默认使用 Optest collector 相同的非 Graph Triton Event 协议。
`do_bench` 使用 25 ms 预热、100 ms 测量和 median。原地操作使用 5 次
预热和 25 个 Event 样本，每次样本前在计时区外恢复实际改写的输入。
有可比原生实现时交替测量三轮，保留双方逐轮耗时并各取中位数。
`--benchmark-samples 1` 可用于筛选。`--benchmark-profile long` 使用
1000 ms 预热、100 ms 测量和三轮配对。`--warmup` 和 `--iter` 改变
非 Graph 时间预算，报告保留实际协议。

`--baseline auto` 在 NVIDIA 运行等价的 OpenCV CUDA 比较器，并在
[已验证的 HIP 用例](hip-validation.md#python-原生比较覆盖)使用 ATen `out=`。
同设备结果使用 `latency`、`latency_base` 和 `speedup`。
`--base-data PATH` 读入 Optest 数据，用独立 shape key 定位记录后核对
完整配置、dtype 和计时协议。同尺寸但 flip code、mask 或布局不同的记录
拥有各自的配置与标识。

正式评价优先采用成功的同设备等价原生比较。原生运行失败时保留失败原因，
可比的外部结果可供展示。没有等价原生实现时，按实际资源和精度核对外部
基线 GPU、系数分母和目标芯片测量。串行 kernel 的利用率按执行时间加权。
`--target-profile PATH` 可提供以 Case ID 索引的实测资源记录，在
`cases[case_id]` 内保存 `vendor`、`chip`、`config`、`timing_protocol`
和带持续时间的 `per_kernel` 利用率。计算资源还要求记录
`resource_precision`。系数为目标设备实测资源值除以实际 NVIDIA 基线
设备实测资源值。外部原始比值除以该系数得到 `compared_speedup`。
缺项保留 `base_miss_reason` 和可比的原始比值。汇总保留原生、等效、仅
原始比值、缺失、不可比和失败的状态与分母。
`tools/check_merge.py` 转发 `--base-data` 和 `--target-profile`，在
`acceptance.json` 的 `python_comparisons` 写入逐例正式状态计数和按比较来源
分组的比值与分母。C++ 原生状态另记为 `cpp_native_statuses`。

原生 0.90×、普通等效 80% 和关键算子等效 95% 用于评审差距和优先级。
项目验收以正确性和报告完整性为门槛；性能数值只用于参考。

## C++ API

`FLAGCV_BUILD_BENCHMARKS=ON` 独立构建 benchmark。每例预热编译缓存，
完成 OpenCV CPU 正确性检查，然后使用预分配缓冲区和同一个原生 stream。

- Device Latency 使用 stream 上的设备 Event 覆盖单次执行。
- API Latency 从 Host 单调时钟上的公开调用前计至该 stream 该次工作完成。
  每次采样前排空该 stream。
- 传入 `--diagnostics` 可同时采集 Host submit 和 batch。
- `add`、`bitwise_not` 和 `flip` 的代表性小用例在同一 stream 对比两种公开接口。

C++ 可报告绝对延迟、同设备历史版本对照及已验证等价的同设备原生对照。
与改动前版本比较时使用同设备、用例、输入域和计时协议。Python 外部
基线的资源系数仅用于 Python kernel 报告。

构建和证据命令见 [测试](testing.md)、
[NVIDIA 验收](nvidia-validation.md) 和 [HIP 验收](hip-validation.md)。
