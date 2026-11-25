# FlagCV 合入标准

本仓库以独立的 CMake 库、公开 C++ API、Triton kernel 和验证工具交付。OpenCV
模块合入遵循其上游流程。FlagCV PR 还需通过下列检查，并在 PR 附上对应证据。
[OpenCV 贡献流程](https://github.com/opencv/opencv/wiki/How_to_contribute)

| 检查 | FlagCV PR 的验收证据 |
| --- | --- |
| 接口和支持范围 | 更新算子清单，列出 dtype、通道、布局、参数默认值、掩码和原地语义。解释与 OpenCV CPU 可观察的差异，保留已有调用方式。 |
| 公共 Runtime 和平台边界 | C++ API 经 lowering、registry、独立 compiler 和 backend。公版可由上游 Triton 编译。厂商扩展或专用编译选项放在相应 backend。 |
| 执行路径 | 公开 API 的设备测试通过。证据涵盖冷编译与稳态复用、默认/原生 stream、输入保护、ROI 和目标写入范围。未注册时使用公版，已注册实现出错时将错误返回调用方。 |
| 正确性 | 用 OpenCV CPU 比较所有声明的路径。整数和位操作逐位一致。浮点使用算子阈值。需要 ±1 LSB 时逐个模式说明原因、误差分布和用例。 |
| 性能 | 分别提交 Python kernel 和 C++ API 的同设备逐用例结果。验证通过后再计时。比较现有发布基线和当前主分支，附原始结果、GPU、工具链、workload 与计时方式。 |
| 安装与文档 | 无 GPU 的 host CI 通过。安装包能由树外 `find_package(FlagCV)` 使用。更新 API 文档、复跑命令和算子清单。 |
| 多平台声明 | 修改已交付的平台需复跑相应设备测试；新增平台提交 SDK/编译器、绑定和 stream、缓存、正确性及性能证据。预览状态写入支持清单。 |

## 性能判定

NVIDIA 同设备 OpenCV CUDA 对照的逐用例目标为 `OpenCV / FlagCV ≥ 0.90`。
Python 和 C++ 独立判定。采样异常时保留首轮证据，隔离 GPU 并重新跑同一套
workload。变更已交付算子时，在相同环境和模式下相对主分支的每例中位延迟
增幅应控制在 5% 以内。超出目标的用例需要在 PR 中记录准确的 workload、测量
结果、原因、影响、负责人和修复条件，由维护者显式评审。逐例差异随结果保留。

其他 GPU 使用同设备厂商 CV 库。缺少原生实现时，可用本机实测带宽或吞吐和
经审阅的读写量或运算量给出等效目标。带宽用同数据类型、尺寸和访问模式的
device-to-device 基线，按实际访问字节数和缓存驻留级别选取对照，记录单位、
计算公式和采样。等效目标计算与性能日志关联到同一用例 ID。

评审时运行 [NVIDIA 复跑说明](docs/nvidia-validation.md)，核对 [算子清单](docs/operators.md)、
[数值容差](docs/numerical-tolerance.md) 和 [接入步骤](docs/operator-development.md)。
