# 测试指南

## Host 测试

`tools/check_merge.py` 构建生产库、运行适用测试并验证迁移后的树外安装包。
命令、测试 XML、原始结果和简要状态写入指定的新目录。Host profile 无需厂商
SDK，但需 C++ 构建工具、支持 CPU 执行的 PyTorch，以及可导入的 `cv2` 和 pytest。

```sh
python tools/check_merge.py --profile host --output /tmp/flagcv-host
cmake -S . -B build-library -DFLAGCV_BACKENDS= \
  -DFLAGCV_BUILD_TESTS=OFF -DFLAGCV_BUILD_BENCHMARKS=OFF
cmake --build build-library --parallel
```

## 设备测试

以下开关都默认 `OFF`。构建时设 `FLAGCV_BUILD_TESTS=ON`，选择相应的
`FLAGCV_BACKENDS`、SDK 和 FlagTree，再打开对应开关。

| 后端 | CMake 开关 | SDK | CTest label |
| --- | --- | --- | --- |
| `nvidia` | `FLAGCV_TEST_NVIDIA` | CUDA Toolkit | `nvidia` |
| `amd` | `FLAGCV_TEST_AMD` | ROCm | `amd` |
| `hygon` | `FLAGCV_TEST_HYGON` | DTK | `hygon` |

配置和复跑见 [NVIDIA 指南](nvidia-validation.md) 或 [AMD/Hygon 指南](hip-validation.md)。
C++ 功能测试使用 OpenCV 5 core；按改动覆盖公开 API、默认与原生 stream、ROI、
mask、原地操作、输入保护和资源释放。若设备绑定有变更且有多卡，检查 SDK 设备
恢复与 `setDevice()` 切换。Python 需要目标后端环境及可导入的 `cv2`：

```sh
python -m pytest tests/test_*.py --builtin --device cuda:0
```

候选注入、独立参考和 profiling 见[算子开发指南](operator-development.md#内部评测与候选注入)。

## 安装与结果范围

安装相关变更应将前缀迁移后从树外构建 [consumer](../tests/consumer)，确认
运行时使用迁移前缀的库、插件和编译数据。设备路径用所选后端的实际 SDK，
验证显式 Handle 和便利接口。改变编译依赖时，检查迁移包中修改共享依赖并重启
进程后能重新生成产物。

保存每个设备、工具链和候选提交对应的结果。缺少实卡时在验证表中写明
待复验项目；离线编译或 Host 测试不等于设备验证。数值规则见
[数值验收](numerical-tolerance.md)，计时见[性能测试](benchmark.md)。
