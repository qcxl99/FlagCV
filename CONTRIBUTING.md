# Contributing to FlagCV

欢迎提交缺陷修复、算子和平台改进。FlagCV 以独立 CMake 库、公开 C++ API、
Triton kernel 和验证工具交付。

## 贡献流程

1. 从主分支创建专题分支。缺陷修复记录复现方式，接口或平台变更先说明预期语义和支持范围。
2. 按 [算子开发指南](docs/operator-development.md) 复用已有 lowering、registry 和公版 Triton kernel。厂商扩展及专用编译选项放在对应 backend。
3. 运行 Host 检查及受改动影响的目标平台测试，更新 [支持清单](docs/operators.md)、接口说明和复现命令。
4. 提交 PR 时附改动目的、受影响范围、实际测试命令、设备与工具链、结果链接及缺项。OpenCV 模块的合入遵循 [OpenCV 贡献流程](https://github.com/opencv/opencv/wiki/How_to_contribute)。

## 按改动范围验收

所有 PR 运行 [Host CI](.github/workflows/host.yml)。在本地可执行
`python tools/check_merge.py --profile host --output /path/to/new-host-evidence`，
覆盖无厂商 SDK 的构建、host 测试和迁移后的树外安装包消费。

| 改动范围 | 设备验收与评审重点 |
| --- | --- |
| 优化已有算子 | 用受影响平台的公开 API 和 Python 入口验证声明路径与边界。比较同设备、同用例、同测量协议的改动前后数据，保留逐例结果和回退理由。 |
| 新增算子或扩大支持范围 | 补公开声明、lowering、registry／provider、独立参考和两种入口的实现及测试。更新算子清单及正确性、性能用例。更新 `workloads/` 和 Optest 元数据。 |
| 公共 Runtime、Compiler、Cache 或打包 | 按受影响的平台和算子声明验证公开 API 及树外安装。覆盖首次编译、复用与应重新编译的条件、缓存隔离、失败传播、默认与外部 stream、调用方资源和稳态同步边界。影响设备绑定时补多卡正反例。 |
| 新增或调整 Backend | 标识实测 SDK、编译器及设备。验证驱动加载和释放、设备和 stream、编译产物、平台实现优先与公版选择、已注册路径出错、树外使用及实际可用的性能对照。缺少实卡证据时保留预览状态。 |

按 [数值容差规范](docs/numerical-tolerance.md) 与 OpenCV CPU 比较受影响的声明路径。明确整数和位操作的逐位要求、浮点阈值及参考例外。测量前确认正确性，按 [性能测试规范](docs/benchmark.md) 分别报告 Python kernel 和 C++ API 的同设备逐用例结果，保留原始数据、用例、同步边界和可比性限制。

共用 kernel 时优先复用符合公开语义的可移植 Triton 实现。记录来源版本和本地
适配，确认签名、dtype、布局、边界与数值一致。FlagCV 通过自身 registry 和安装包
定位 kernel，各交付项目分别验收公开接口、生产执行路径和性能。

评审提交范围、源码身份、环境、执行日志、逐例结果和缺项说明。每个适用项都要能
追到实际证据。单次 NVIDIA 验收只覆盖该设备、当前用例及执行路径，新增能力和其他
平台需附相应验证。安装或功能失败、必要结果缺失应保留为待修，不写成通过。

编译器、缓存或 Backend 变更时，保留直接证明分派、加载、复用与释放的
测试证据。测试命令、协议和结果标准见 [功能验收](docs/testing.md)、
[性能测试](docs/benchmark.md) 和 [算子清单](docs/operators.md)。
