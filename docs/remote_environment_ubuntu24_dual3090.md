# Ubuntu 24.04 双 RTX 3090 环境配置

本文用于在全新 Ubuntu 24.04 x86_64 服务器上部署 Mamba-DEKF。配置以仓库当前的 `config/train_nuscenes.yaml` 为准：训练主干为 Mamba-2，运行时要求真实 Mamba SSM，不允许静默回退到 GRU。

## 1. 推荐软件组合

| 组件 | 版本/选择 | 说明 |
|---|---|---|
| OS | Ubuntu 24.04 LTS x86_64 | NVIDIA 驱动通过发行版包管理器或管理员统一安装 |
| Python | 3.10 | 使用 Conda/Miniforge 环境，避免和 Ubuntu 系统 Python 3.12 混用 |
| PyTorch | 2.1.0 + CUDA 12.1 | 与仓库已有 Mamba 源码安装记录一致 |
| torchvision | 0.16.0 | 与 PyTorch 2.1.0 配套 |
| mamba-ssm | v2.2.4 源码构建 | 为当前 Mamba-2 配置构建 CUDA 扩展 |
| CUDA Toolkit | 12.1 | 需要 `nvcc` 编译 Mamba 扩展；PyTorch wheel 自带 CUDA runtime |
| C/C++ 编译器 | GCC/G++ 12 | Ubuntu 24.04 默认编译器可能高于该版本，编译 CUDA 扩展时显式指定 GCC 12 |

项目代码支持 Mamba-1 和 Mamba-2，但当前训练 YAML 的 `MODEL.BACKBONE` 是 `mamba2`。README 中旧的 `mamba-ssm>=1.2.0` 安装示例不适用于当前训练配置；以本文固定的 `v2.2.4` 为准。`requirements.txt` 中的宽松版本下限也可能让 pip 选取不同版本，因此源码安装完成后不要再运行会升级 `mamba-ssm` 的安装命令。

## 2. 主机与 NVIDIA 驱动

确认系统和两张卡可见：

```bash
lsb_release -a
uname -m
nvidia-smi
```

`nvidia-smi` 应列出两张 RTX 3090。若命令不存在或没有 GPU，由管理员先安装与服务器内核兼容的 NVIDIA 专有驱动并重启。不要用 pip 安装 NVIDIA 驱动。PyTorch 的 `torch.version.cuda` 表示 wheel 使用的 CUDA runtime 版本，不要求系统 CUDA Toolkit 与其完全由同一安装器提供；但编译 Mamba 时需要可用的 CUDA Toolkit 和 `nvcc`。

安装基础工具（需要 sudo）：

```bash
sudo apt update
sudo apt install -y build-essential gcc-12 g++-12 git wget curl ca-certificates \
  ninja-build pkg-config
```

如果系统仓库没有 `gcc-12`/`g++-12`，使用 Ubuntu toolchain PPA 或管理员提供的编译器，不要直接以 Ubuntu 24.04 默认 GCC 版本编译失败后反复重装 Python 包。

Ubuntu 24.04 上建议把 CUDA 12.1 Toolkit 放在后续创建的 Conda 环境中，避免混用旧版系统级 CUDA 仓库。Toolkit 安装命令见第 3 节。若选择系统级 Toolkit，按 NVIDIA 对应版本的安装指南配置，并把后续的 `CUDA_HOME` 改为实际安装路径。

## 3. 创建 Python 环境

以下以 Miniforge/Conda 已安装为前提：

```bash
conda create -n mamba-dekf python=3.10 -y
conda activate mamba-dekf

conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
python -m pip install --upgrade pip setuptools wheel packaging ninja
nvcc --version
test -f "$CONDA_PREFIX/include/cuda_runtime.h" && echo "CUDA headers found"

git clone https://github.com/Papagan/Mamba_DEKF.git ~/Mamba-DEKF
cd ~/Mamba-DEKF
```

安装 PyTorch CUDA 12.1 wheel：

```bash
python -m pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

确认 PyTorch 能识别两张卡：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

预期 `torch.version.cuda` 为 `12.1`、`CUDA available` 为 `True`、GPU count 为 `2`。

## 4. 安装项目依赖

在仓库根目录运行：

```bash
python -m pip install "numpy>=1.22,<2"
# 保留已固定的 PyTorch/CUDA 与稍后从源码安装的 Mamba-2，不让宽松约束覆盖它们
grep -Ev '^(torch|mamba-ssm)' requirements.txt > /tmp/mamba-dekf-requirements.txt
python -m pip install -r /tmp/mamba-dekf-requirements.txt
python -m pip install tensorboard
```

`numpy<2` 用于减少项目中的旧版数据处理/评估依赖与 NumPy 2.x 的兼容问题。训练入口直接导入 TensorBoard 的 `SummaryWriter`，而它未单独列在当前 `requirements.txt`，因此需要额外安装。

## 5. 编译安装 Mamba-2

源码编译前确认 PyTorch、CUDA Toolkit 和编译器来自预期环境：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export MAX_JOBS=4

nvcc --version
"$CC" --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
```

构建仓库记录的 Mamba `v2.2.4`：

```bash
mkdir -p ~/src
git clone https://github.com/state-spaces/mamba.git ~/src/mamba
cd ~/src/mamba
git checkout v2.2.4
python -m pip install . --no-build-isolation --no-cache-dir -v
cd -
```

`--no-build-isolation` 可确保扩展使用当前环境中的 CUDA 版 PyTorch 头文件。首次编译可能需要数分钟。若显存/主机内存或并行编译资源紧张，可将 `MAX_JOBS` 调为 `2`。

验证 Mamba-2 的 CUDA 前向：

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba2

assert torch.cuda.is_available()
model = Mamba2(
    d_model=96,
    d_state=64,
    d_conv=4,
    expand=2,
    headdim=64,
    ngroups=1,
).cuda()
x = torch.randn(2, 8, 96, device="cuda")
y = model(x)
torch.cuda.synchronize()
print("Mamba2 output:", tuple(y.shape))
PY
```

预期输出形状为 `(2, 8, 96)`。

## 6. 验证项目配置

检查训练配置要求的主干：

```bash
rg -n "BACKBONE:|D_MODEL:|D_STATE:|MAMBA2_HEADDIM:|REQUIRE_MAMBA_SSM:" \
  config/train_nuscenes.yaml
```

执行最小项目验证：

```bash
python tools/verify_pipeline.py
```

训练日志开头应报告 `backbone=mamba2`。若显示 `fallback_gru`，不要开始正式训练；配置设置了 `REQUIRE_MAMBA_SSM: true`，训练入口应直接报错。检查以下导入与 CUDA 扩展：

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba2
from kalmanfilter.mamba_adaptive_kf import Mamba1, Mamba2 as ProjectMamba2

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Mamba2 package:", Mamba2)
print("project Mamba1 available:", Mamba1 is not None)
print("project Mamba2 available:", ProjectMamba2 is not None)
print("GPUs:", torch.cuda.device_count())
PY
```

## 7. 双卡使用说明

当前 `training/train.py` 默认使用 `cuda`，但没有实现 `DistributedDataParallel`、`DataParallel` 或双进程启动逻辑。双 3090 环境下，该训练脚本默认只会使用一张卡；两张卡的显存不会自动合并。可用 `CUDA_VISIBLE_DEVICES=1` 指定单卡：

```bash
CUDA_VISIBLE_DEVICES=1 python training/train.py --config config/train_nuscenes.yaml
```

训练命令：

```bash
python training/train.py --config config/train_nuscenes.yaml
```

PyTorch CUDA wheel 主要提供运行时；Mamba-2 扩展在运行时应使用当前进程选中的 GPU。若后续要并行训练，需要额外实现并验证 DDP，不能仅把启动命令改成 `torchrun`。

## 8. nuScenes 数据与路径

环境安装不包含 nuScenes 数据、CenterPoint 检测结果或训练缓存。按服务器目录修改 `config/train_nuscenes.yaml` 里的路径，至少需要：

- nuScenes 数据目录，包含 `samples/`、`sweeps/`、`maps/` 和 `v1.0-trainval/`
- 与当前训练流程对应的 CenterPoint `val.json`
- `TRAIN_TRACKLET_PKL`、`VAL_TRACKLET_PKL` 指向的 fusion tracklet 缓存
- Stage B 启用时，`TRAIN_PAIRWISE_PKL`、`VAL_PAIRWISE_PKL` 指向 pairwise association 缓存

仓库 README 中的 `/root/autodl-tmp/...` 是原运行环境示例路径，远程 Ubuntu 机器上必须替换。训练配置还引用一个 inference YAML；该文件路径也必须从仓库根目录可解析。

## 9. 训练前检查清单

```bash
conda activate mamba-dekf
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "from mamba_ssm import Mamba2; print('Mamba2 import OK')"
python tools/verify_pipeline.py
```

确认数据和缓存路径可读写后启动训练，并检查日志中的设备与主干：

```bash
python training/train.py --config config/train_nuscenes.yaml
```

训练结果默认写入配置的 `TRAINING.SAVE_DIR`（当前为 `checkpoints/mamba_motion/`）。长时间远程任务建议在 `tmux`/`screen` 中运行，并将代码、配置、缓存与 checkpoint 定期备份到持久化存储。

## 10. 常见问题

### PyTorch 显示 CUDA 不可用

先确认 `nvidia-smi` 正常，再确认安装的是 `cu121` wheel，而不是 CPU wheel。`nvcc` 不影响 PyTorch wheel 的基本 CUDA 检测，但 Mamba 源码扩展编译需要 Toolkit。

### 找不到 `cuda_runtime.h` 或 `nvcc`

安装 CUDA Toolkit 12.1，并设置正确的 `CUDA_HOME`、`PATH`。仅安装 NVIDIA 驱动不包含编译器和头文件。

### Mamba 扩展编译器报错

检查 `CC`、`CXX`、`CUDAHOSTCXX` 是否指向 GCC 12；检查 `torch.version.cuda` 和 `nvcc --version` 是否都属于 CUDA 12.1 构建组合；把 `MAX_JOBS` 降低后重新编译。

### `undefined symbol` 或 import 二进制扩展失败

通常是 Mamba 扩展与当前 PyTorch/CUDA ABI 不匹配。确认正在使用 `mamba-dekf` 环境，并在该环境中从 `v2.2.4` 重新构建 Mamba，避免复用其他 Python 环境编译出的 `.so` 文件。

### 训练退回 GRU 或报 backbone 错误

当前配置要求 `mamba2`。确认 `from mamba_ssm import Mamba2` 成功、项目导入的 `Mamba2` 不为 `None`，且 `config/train_nuscenes.yaml` 没有被改成其他主干。

## 官方参考

- [PyTorch previous versions: 2.1.0 / CUDA 12.1](https://docs.pytorch.org/get-started/previous-versions/)
- [Mamba 官方仓库及安装说明](https://github.com/state-spaces/mamba)
- [Mamba v2.2.4 release](https://github.com/state-spaces/mamba/releases/tag/v2.2.4)
- [NVIDIA CUDA Toolkit 12.1 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/)
