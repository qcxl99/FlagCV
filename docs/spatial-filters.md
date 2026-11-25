# GaussianBlur 与 filter2D

两种接口接受调用方的输入、输出和系数缓冲区。每个 GpuMat 是独立有效区域。
输入和输出可以有不同的字节行距。两者的像素不能重叠。卷积邻域只读取
视图内的像素，等同于 OpenCV 的 BORDER_ISOLATED。边界支持 CONSTANT（零）、
REPLICATE、REFLECT 与 REFLECT_101。调用在给定原生流上提交。

## 准备与执行

- 构造 GaussianPlan 或 Filter2DPlan。将 table() 返回的字节上传到 double 对齐的
  设备分配。保留计划直到上传完成。
- 传入 DeviceWorkspace。保持系数只读且与图像分离，排序上传与所有读取流。
  保持设备分配及流直到最后一次调用完成。
- GaussianBlur 的两 pass 重载另外接受可写 GpuMat。尺寸和通道与输入相同，
  深度由 scratch_depth() 返回。中间图与输入、输出和系数表分离。
  两个 kernel 在同一流内顺序执行。调用方管理中间图和全部分配。

GaussianBlur 支持 U8、U16、F32、F64 与 1–4 个交错通道。尺寸可显式指定为
正奇数，也可在 sigma 为正时通过非正轴尺寸启用自动推导。
非正的 sigma_y 继承 sigma_x。非 CONSTANT 边界模式的单像素轴可折叠。
轴宽最大 65535，较大的核仍走通用循环。
U8/U16 每个轴分别量化为 8/16 位小数，使用误差扩散和中心残差保持归一。
U16 的横向累加用 U32，纵向用 U64。整数仅在末端执行半向上舍入。

filter2D 做相关。系数按行排列，默认锚点为核中心，两轴可以独立指定。
输出深度默认为源深度。U8 可以输出 U8、U16、S16、F32 或 F64。
U16 和 S16 各可输出同深度、F32 或 F64。F32 与 F64 也各自支持同深度。
输出为 F64 时用 F64 工作精度，其余用 F32。
先加入 delta，再按行顺序累加非零权重，禁用乘加融合。
整数输出先按最近偶数转换到有符号 i32，再饱和到目标深度。
固定 SSE3 OpenCV 的无效 i32 转换结果为 INT_MIN。
全零核保留左上角的零 tap，使该位置的非有限输入按固定 CPU 传播。

## 数值范围与性能验证

固定参考是 OpenCV 5.0.0 SSE3 的 CPU 实现，分别用 CPU 输出核对准备系数和 GPU 执行。
整数逐字节比较。F32/F64 有限输入测试使用逐元素误差上限
2e-5 × max(1, abs(ref)) 和 4e-12 × max(1, abs(ref))。
通用较大相关核用直接循环。OpenCV 的 DFT 路径可能有不同的舍入顺序，
因此需要按较大核、输入域和深度分别验证。

Gaussian 的小核可用单次提交。较大核的两 pass 路径由调用方提供中间图。
C++ 计时包括所有提交、同步及中间图流量，分配与表上传单计。
在 gfx936 上，实测 F32 C3 相关的 128 元素 tile 使用两个 wave。
通用 kernel 和其它类型或平台仍由原计划选择配置。

HIP 测试目标是 flagcv.hip.spatial_filters。其测试程序还提供
--gaussian-boundary 和 --benchmark，分别复验舍入边界与选定公开调用的同步 wall。
