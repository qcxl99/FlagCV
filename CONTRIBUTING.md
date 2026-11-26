# FlagCV 合入标准

本仓库以独立的 CMake 库、公开 C++ API、Triton kernel 和验证工具交付。OpenCV
模块合入遵循其上游流程。FlagCV PR 还需通过下列检查，并在 PR 附上对应证据。
[OpenCV 贡献流程](https://github.com/opencv/opencv/wiki/How_to_contribute)

| 检查 | FlagCV PR 的验收证据 |
| --- | --- |
| 接口和支持范围 | 更新算子清单，列出 dtype、通道、布局、参数默认值、掩码和原地语义。解释与 OpenCV CPU 可观察的差异，保留已有调用方式。 |
| 公共 Runtime 和平台边界 | C++ API 经 lowering、registry、独立 compiler 和 backend。交付使用对应平台的 FlagTree。厂商扩展或专用编译选项放在相应 backend。 |
| 执行路径 | 公开 API 的设备测试通过。证据涵盖冷编译与稳态复用、默认/原生 stream、输入保护、ROI 和目标写入范围。未注册时使用公版，已注册实现出错时将错误返回调用方。 |
| 正确性 | 用 OpenCV CPU 比较所有声明的路径。整数和位操作逐位一致。浮点使用算子阈值。已确认的参考实现差异需列出精确范围、正式参考和回归用例。需要 ±1 LSB 时逐个模式说明原因、误差分布和用例。 |
| 性能 | 分别提交 Python kernel 和 C++ API 的同设备逐用例结果。验证通过后再计时。有可比的发布或主分支数据时附逐例差异，并保留原始结果、GPU、工具链、workload 与计时方式。 |
| 安装与文档 | 无 GPU 的 host CI 通过。安装包能由树外 `find_package(FlagCV)` 使用。更新 API 文档、复跑命令和算子清单。 |
| 多平台声明 | 修改已交付的平台需复跑相应设备测试；新增平台提交 SDK/编译器、绑定和 stream、缓存、正确性及性能证据。预览状态写入支持清单。 |

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

共用 kernel 时优先复用符合公开语义的可移植 Triton 实现。记录来源版本和本地
适配，确认签名、dtype、布局、边界与数值一致。FlagCV 通过自身 registry 和安装包
定位 kernel，各交付项目分别验收公开接口、生产执行路径和性能。

评审提交范围、源码身份、环境、执行日志、逐例结果和缺项说明。每个适用项都要能
追到实际证据。单次 NVIDIA 验收只覆盖该设备、当前用例及执行路径，新增能力和其他
平台需附相应验证。安装或功能失败、必要结果缺失应保留为待修，不写成通过。

编译器、缓存或 Backend 变更时，保留直接证明分派、加载、复用与释放的
测试证据。测试命令、协议和结果标准见 [功能验收](docs/testing.md)、
[性能测试](docs/benchmark.md) 和 [算子清单](docs/operators.md)。
