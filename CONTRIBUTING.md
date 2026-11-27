# Contributing to FlagCV

## 开发

从主分支创建专题分支，说明问题、预期语义和受影响的平台。复用现有 lowering、
registry 和公版 Triton kernel，厂商专用逻辑放在对应 backend。复用外部 kernel
时记录来源版本、本地适配及语义核对。参见[算子开发指南](docs/operator-development.md)。

## 代码风格

Python 用 `snake_case` 函数和变量名、`UPPER_CASE` 常量名；可用仓库的
[Flake8 配置](.flake8)做本地检查。C++ 使用 C++17，沿用相邻文件风格：
公开类名和类型名用 `PascalCase`，现有算子名与 OpenCV 风格 API 保持一致，
内部 helper 通常用 `snake_case`。启用构建警告并优先维持现有接口约定。

## 测试和性能

[Host CI](.github/workflows/host.yml)在 PR 上运行。可用
`python tools/check_merge.py --profile host --output /path/to/new-host-evidence`
复跑无厂商 SDK 的测试和迁移安装消费；依变更范围补 [设备测试](docs/testing.md)：

| 变更范围 | 必要验收 |
| --- | --- |
| 文档 | 核对链接、锚点和示例与源码；纯文档改动不需要设备报告。 |
| 算子新增、修复或优化 | 覆盖声明的 C++ 和内部 Python 路径及边界，与 OpenCV CPU 核对；新增用例同步 workload、Optest 元数据和 [算子清单](docs/operators.md)。影响性能时附同卡、同协议前后对照。 |
| Runtime、Compiler、Cache 或打包 | 按影响范围验证缓存首次编译、复用或失效、分派及错误传播、调用方资源及迁移安装；涉及设备绑定时覆盖多卡正反例。 |
| Backend | 记录实际 SDK、编译器和设备；验证产物 ABI、平台优先与公版分派、已注册实现报错、设备/stream 生命周期及树外运行。缺设备验证时标明待复验。 |

先按[数值规则](docs/numerical-tolerance.md)验证正确性，再按[计时协议](docs/benchmark.md)
区分 Python kernel 与 C++ API 性能，说明输入域、比较来源和不可比项。

## Pull Request

用 PR 模板列出目标、改动、实际测试命令与结果。普通 PR 只需提供与改动相称的
验证，不必附全套验收报告；扩大支持范围或改变公开执行路径时，链接对应候选、设备和
工具链的证据。标记未验证或不适用项，不能把缺项写成通过。涉及 OpenCV 自身模块的
提交遵循 [OpenCV 贡献流程](https://github.com/opencv/opencv/wiki/How_to_contribute)。
