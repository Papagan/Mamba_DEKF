# NumPy 1.x 编译扩展与 NumPy 2.2.6 的兼容性

## 已定位的远程环境（numpy_enviroment.json）

报告显示 Python 3.10.22、NumPy 2.2.6、Torch 2.13.0+cu132。
Torch↔NumPy 转换、SciPy、OpenCV、lap、pandas、Numba 和 Mamba 导入探针通过；尚不代表 CUDA 内核测试通过。
失败有两处：Matplotlib 3.5.3 的 `_path` 扩展出现 NumPy ABI 错误，连带导致 nuScenes 导入失败；
motmetrics 1.1.3 从 `collections` 导入 `Iterable`，不兼容 Python 3.10。

这里需要区分 SDK 版本：[1.1.11 的元数据](https://pypi.org/pypi/nuscenes-devkit/1.1.11/json)
限制 Matplotlib `<3.6`，但没有限制 NumPy `<2`，因此 `pip check` 可以通过而实际导入失败。
[OpenCV 5.0.0.93](https://pypi.org/pypi/opencv-python/5.0.0.93/json) 在 Python >=3.9 下要求 NumPy >=2，
不能仅降低 NumPy。仓库提供针对这个 Python 3.10 环境的协调修复组合，保留已安装的 Torch/CUDA/Mamba：

```bash
conda activate mamba-dekf
python -m pip freeze > numpy-repair-before.txt
python -m pip install --dry-run --upgrade -r requirements-nuscenes-py310.txt
python -m pip install --upgrade -r requirements-nuscenes-py310.txt
python tools/check_numpy_environment.py --output numpy_environment_after.json
python tools/verify_pipeline.py --device cuda:0 --require-cuda
python tools/verify_pipeline.py --device cuda:1 --require-cuda
```

该组合将 NumPy 改为 1.26.4、OpenCV 改为 4.11.0.86、motmetrics 改为 1.4.0；
保留 nuScenes 1.1.11 和 Matplotlib 3.5.3。没有远程执行安装，完成后以上述诊断和 GPU 预检为准。
`numpy-repair-before.txt` 仅是安装前版本清单，不是完整环境备份。
后续安装通用依赖时加 `-c requirements-nuscenes-py310.txt`，避免又被升级回冲突组合。
如果必须保留 NumPy 2.2.6，则需另选或维护兼容的 SDK/评估依赖，而非只升级 Matplotlib。

## 通用诊断

`A module that was compiled using NumPy 1.x cannot be run in NumPy 2...` 指向依赖扩展的二进制 ABI，
不是模型权重格式。仅修改 Python 中的 NumPy 函数、忽略警告或更换 checkpoint 都不能修复它。
[NumPy 官方说明](https://numpy.org/doc/stable/user/troubleshooting-importerror.html#downstream-importerror-attributeerror-or-c-api-abi-incompatibility)
建议从堆栈中定位具体扩展，升级或重新编译该扩展；无法升级时才考虑兼容的 NumPy 1.x 环境。

在发生报错的同一个远程 Conda 环境、同一个项目目录执行：

```bash
python tools/check_numpy_environment.py --output numpy_environment.json
```

脚本不安装或修改任何包，只逐个在独立进程测试 NumPy、PyTorch、SciPy、OpenCV、lap、pandas、
Matplotlib、Numba、nuScenes、motmetrics 和 Mamba 导入/基本运算，保存版本、Python 路径和完整异常。
PyTorch 额外测试 `torch.from_numpy()` 和 `.numpy()` 双向转换，避免“import 成功但训练时崩溃”。
即使进程返回 0，出现 ABI 警告也判定为失败。最后执行 `pip check` 检查声明的依赖冲突。
只查运动训练关键依赖可加 `--modules numpy torch scipy cv2 mamba_ssm`；`pip check` 仍检查整个环境。
Mamba 探针只检查导入，CUDA 内核仍需用下面的预检验证。

若希望保留 NumPy 2.2.6，必须为报告中失败的扩展安装支持 NumPy 2 的构建。
不要在不知道失败包的情况下整体升级 Torch/CUDA/Mamba；它们之间也存在二进制兼容关系。
若需要更换 Torch，应随后核对、重装或重编译与之配套的 Mamba/causal-conv1d，再运行：

```bash
python tools/verify_pipeline.py --device cuda:0 --require-cuda
python tools/verify_pipeline.py --device cuda:1 --require-cuda
```

nuScenes 全链路还有独立约束：核查时[官方 SDK requirements](https://github.com/nutonomy/nuscenes-devkit/blob/master/setup/requirements/requirements_base.txt)
明确要求 `numpy>=1.22.0,<2.0.0`。因此本仓库源码接口兼容 NumPy 2，并不代表官方 SDK 和所有二进制依赖
都支持它。完整生成/评估流程应优先使用满足 SDK 约束的隔离环境；如果坚持 NumPy 2，需验证所选 SDK
版本/分支及评估指标，不应通过 `--no-deps` 或全局猴子补丁强行跳过依赖约束。
Python 3.14 环境不能直接照搬旧 Python 的 NumPy 1.x 安装方案；先确认诊断报告中的实际 Python 版本。

本次源码调整替换了 `np.mat`、`np.float_` 和 vendored TrackEval 的旧 `np.float/int/bool` 别名。
`np.mat` 改为 `np.asmatrix`，保留原来的矩阵乘法语义；MCTrack 滤波数学、模型和 checkpoint 不变。
这些源码修改本身不要求重新生成数据或重新训练，但仍不能替代失败扩展的 ABI 修复。
