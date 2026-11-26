# 功能验收

`tools/check_merge.py` 构建生产库、运行适用测试并验证迁移后的树外安装包。
命令、测试 XML、原始结果和简要验收状态写入指定目录。

```sh
python tools/check_merge.py --profile host --output /tmp/flagcv-host
cmake -S . -B build-library -DFLAGCV_BACKENDS= \
  -DFLAGCV_BUILD_TESTS=OFF -DFLAGCV_BUILD_BENCHMARKS=OFF
cmake --build build-library --parallel
```

GPU 验收通过选择 `FLAGCV_BACKENDS` 和对应的 `FLAGCV_TEST_<VENDOR>`，
验证六算子的公开 C++ 路径、默认及原生 stream、ROI、mask、原地操作、
输入保护和资源释放。双卡可用时验证 SDK 当前设备恢复和 `setDevice()` 切换。
C++ 功能测试使用 OpenCV 5 core。Python 需要可导入的 `cv2`，并保留
`--override` 注入、`--reference` 独立参考和 profiling。

```sh
python -m pytest tests/test_*.py --builtin --device cuda:0
```

安装验证将前缀迁移后从树外构建 `tests/consumer`。测试显式 Handle 和
便利接口，GPU 路径使用所选后端的实际 SDK。改变编译依赖时，在迁移后的
安装包中验证对共享依赖的改动能使重启进程重新生成产物。

保存每个设备、工具链和候选提交对应的结果。缺少实卡时在验证表中写明
待复验项目。数值规则见 [数值容差](numerical-tolerance.md)。
