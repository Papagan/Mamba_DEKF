# Ubuntu 24.04 双 RTX 3090 环境配置（Python 3.14 / Driver 595.99.02）

本文针对远程机已安装 Ubuntu 24.04 x86_64、两张 RTX 3090、NVIDIA Linux 驱动 595.99.02 的情况。目标是运行本仓库当前 `config/train_nuscenes.yaml` 中的 Mamba-2 主干。命令默认使用 Miniforge/Conda 隔离 Python 环境；不要替换 Ubuntu 系统 Python。

## 1. 版本选择与兼容性说明

| 组件 | 本教程选择 | 说明 |
|---|---|---|
| Python | 3.14（Conda 环境） | 不改系统 Python。项目依赖需逐项安装验证 |
| NVIDIA 驱动 | 已有 595.99.02 | 满足 CUDA 13.2 Update 1 的 Linux 驱动最低版本 595.45.04 |
| CUDA Toolkit | 13.2 | 提供 `nvcc`、头文件和编译工具；不会安装/替换驱动 |
| PyTorch | 2.13.0 + CUDA 13.2 wheel | PyTorch 官方提供的安装组合 |
| Mamba-SSM | v2.3.2.post1 源码编译 | 发布资产未列出 CPython 3.14 wheel，需在目标机上编译；Python 3.14 + 此版本尚无本仓库实测记录 |

驱动、Toolkit 和 PyTorch CUDA runtime 是不同组件：`nvidia-smi` 显示驱动可支持的最高 CUDA 版本；`nvcc --version` 显示本机 Toolkit；`torch.version.cuda` 显示 PyTorch wheel 使用的 CUDA runtime。这里三者分别约为 13.2、13.2、13.2。不要为了 PyTorch 再安装 CUDA runtime，也不要安装会连带升级驱动的 `cuda` 或 `cuda-drivers` 元包。

**重要风险提示：** PyTorch 2.13.0 官方提供 cu132 wheel，但 Mamba v2.3.2.post1 的已发布 Linux wheel 最高只到 CPython 3.13。Python 3.14 下的 Mamba 必须源码编译，且要通过本文的 Mamba2 GPU 前向测试后才能开始正式训练。如果编译或测试失败，优先采用文末的 Python 3.10 已验证组合，而不是绕过 CUDA 算子测试。

## 2. 检查主机、GPU 和驱动

```bash
lsb_release -a
uname -m
nvidia-smi
```

`uname -m` 应为 `x86_64`，`nvidia-smi` 应显示两张 RTX 3090 和 Driver Version `595.99.02`。若 GPU 不可见，先让管理员修复驱动或内核模块；不要通过 pip 或 CUDA Toolkit 安装器重装驱动。

安装构建基础工具：

```bash
sudo apt update
sudo apt install -y build-essential gcc g++ git wget curl ca-certificates \
  pkg-config ninja-build
```

CUDA 13.2 支持 Ubuntu 24.04，使用系统默认 GCC 即可。检查：

```bash
gcc --version
g++ --version
```

## 3. 安装 CUDA Toolkit 13.2（提供 nvcc）

以下安装仅 Toolkit，不会主动安装驱动：

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-2
```

配置当前 shell：

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
test -f "$CUDA_HOME/include/cuda_runtime.h" && echo "CUDA headers found"
```

应看到 CUDA 13.2 版本和 `CUDA headers found`。为了每次登录生效，可将上面两行 `export` 加入 `~/.bashrc`，再运行 `source ~/.bashrc`。不要全局追加 CUDA Toolkit 的 `lib64` 到 `LD_LIBRARY_PATH`，以免它覆盖 PyTorch wheel 自带的 CUDA 库。

## 4. 创建 Python 3.14 隔离环境

先在远程机安装 Miniforge/Conda。确认 Conda 能获取 Python 3.14 后创建环境：

```bash
conda create -n mamba-dekf-py314 python=3.14 -y
conda activate mamba-dekf-py314
python --version
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

确认 `python --version` 为 `3.14.x`。如果 Conda channel 尚未提供该版本，可用官方 Python 3.14 安装并建立 venv，但不要覆盖 `/usr/bin/python3`。

## 5. 安装 PyTorch CUDA 13.2

安装 PyTorch 官方列出的配套版本：

```bash
python -m pip install \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu132
```

验证 PyTorch 是否加载 CUDA 并识别两张卡：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit("Expected two CUDA-visible GPUs")
PY
```

预期 `torch.version.cuda` 是 `13.2`、GPU count 是 `2`。若 pip 提示没有匹配 Python 3.14 的 torch wheel，检查 Python 架构是否为 x86_64、pip 是否为最新，以及 PyTorch 官方页面是否仍提供该组合；不要误装 CPU wheel。

## 6. 安装仓库 Python 依赖

进入服务器上的项目目录后执行。不要直接安装未经筛选的 `requirements.txt`，其中宽泛的 `torch>=1.12.0` 和 `mamba-ssm>=2.2.0` 可能改变已选版本或触发不合适的预编译包安装。

```bash
cd ~/Mamba-DEKF
python -m pip install "numpy>=1.22,<2"
grep -Ev '^(torch|mamba-ssm)' requirements.txt > /tmp/mamba-dekf-requirements.txt
python -m pip install -r /tmp/mamba-dekf-requirements.txt
python -m pip install tensorboard
```

`tensorboard` 单独安装是因为训练入口使用 `SummaryWriter`，但它未单独列在当前依赖文件中。若某个依赖报告“不支持 Python 3.14”或没有可用 wheel，不要强行改装系统 Python；先按第 10 节切换到项目的已验证环境。

## 7. 从源码编译 Mamba-SSM

编译前确认当前 Python、PyTorch 与 Toolkit 路径：

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
nvcc --version
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.__file__)"
```

RTX 3090 的计算能力为 8.6。克隆固定版本源码并强制本机编译 CUDA 扩展：

```bash
mkdir -p ~/src
git clone --branch v2.3.2.post1 --depth 1 \
  https://github.com/state-spaces/mamba.git ~/src/mamba
cd ~/src/mamba
MAMBA_FORCE_BUILD=TRUE \
  MAX_JOBS=4 python -m pip install . \
  --no-build-isolation --no-cache-dir -v
cd ~/Mamba-DEKF
```

编译可能持续较久并消耗较多内存；若构建进程被系统 OOM killer 终止，把 `MAX_JOBS` 降到 `2`。`--no-build-isolation` 确保扩展使用当前环境中的 PyTorch。不要在安装 Mamba 后再运行会升级 Torch 或 Mamba 的依赖安装命令。

## 8. 必须通过的 CUDA / Mamba-2 验证

先单独验证 Mamba2 在 GPU 上完成前向计算，而不只是 Python import 成功：

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba2

assert torch.cuda.is_available()
model = Mamba2(
    d_model=96, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1
).cuda()
x = torch.randn(2, 8, 96, device="cuda")
y = model(x)
torch.cuda.synchronize()
print("Mamba2 output:", tuple(y.shape))
assert tuple(y.shape) == (2, 8, 96)
PY
```

预期输出 `Mamba2 output: (2, 8, 96)`。若出现 `no kernel image`、扩展导入失败、undefined symbol 或 CUDA 编译错误，表示该 Python/Torch/CUDA/Mamba 组合尚未成功，先停止，不要开始训练。保留完整的 `pip -v` 错误日志以排查。

再验证项目实际导入和配置：

```bash
cd ~/Mamba-DEKF
python - <<'PY'
import torch
from mamba_ssm import Mamba2
from kalmanfilter.mamba_adaptive_kf import Mamba1, Mamba2 as ProjectMamba2

print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Mamba2 package:", Mamba2)
print("project Mamba1 available:", Mamba1 is not None)
print("project Mamba2 available:", ProjectMamba2 is not None)
print("GPU count:", torch.cuda.device_count())
assert ProjectMamba2 is not None
PY
rg -n "BACKBONE:|D_MODEL:|D_STATE:|MAMBA2_HEADDIM:|REQUIRE_MAMBA_SSM:" \
  config/train_nuscenes.yaml
python tools/verify_pipeline.py
```

当前 YAML 应为 `BACKBONE: mamba2` 且 `REQUIRE_MAMBA_SSM: true`。项目验证成功且日志没有 fallback 到 GRU 后，才开始训练：

```bash
python training/train.py --config config/train_nuscenes.yaml
```

## 9. 双卡说明

当前 `training/train.py` 默认选择一张 `cuda`，没有实现 `DistributedDataParallel` 或双卡训练逻辑。两张 3090 不会自动合并显存，也不会因为 `CUDA_VISIBLE_DEVICES` 同时列出两张卡就自动并行。选择第二张卡运行单卡任务：

```bash
CUDA_VISIBLE_DEVICES=1 python training/train.py --config config/train_nuscenes.yaml
```

要使用双卡并行，需要在项目中实现并验证 DDP，再用 `torchrun` 启动。

## 10. 推荐回退：项目已验证的 Python 3.10 组合

若 Python 3.14 下任一依赖或 Mamba CUDA 前向失败，建议为训练单独创建 Python 3.10 环境。仓库历史安装记录对应 Python 3.10.20、PyTorch 2.1.0+cu121、Mamba v2.2.4；该组合比 3.14 新栈更接近项目原有实测环境。系统驱动 595.99.02 可向后兼容 CUDA 12.1 应用，不必降级驱动。

```bash
conda create -n mamba-dekf-py310 python=3.10 -y
conda activate mamba-dekf-py310
python -m pip install --upgrade pip setuptools wheel packaging ninja
conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

回退环境使用 Conda 内 CUDA Toolkit 12.1，与 cu121 PyTorch 配套；不要让系统的 13.2 `nvcc` 排在该环境的 `PATH` 前面。之后按仓库 [Mamba 源码安装指南](mamba-ssm-source-install.md) 构建 v2.2.4，并完成其 Mamba2 GPU 前向验证。

## 11. 常见问题

### `nvidia-smi` 正常，但 PyTorch 显示 CUDA 不可用

确认安装的是 `cu132` wheel 而不是 CPU wheel，并检查 `torch.version.cuda`。`nvcc` 是否安装不影响 PyTorch wheel 的基础 CUDA 检测。

### 找不到 `nvcc` 或 `cuda_runtime.h`

确认安装 `cuda-toolkit-13-2`，并设置 `CUDA_HOME=/usr/local/cuda-13.2`、`PATH="$CUDA_HOME/bin:$PATH"`。仅 NVIDIA 驱动不包含编译器和开发头文件。

### Mamba 在 Python 3.14 无法编译或运行

目前 v2.3.2.post1 没有 CPython 3.14 预编译 wheel，源码构建属于需要本机验证的路径。先检查完整构建日志、GPU 架构 `8.6` 和编译任务数；如果 CUDA 前向仍未通过，使用第 10 节 Python 3.10 环境。

### 项目静默回退 GRU 或报告 backbone 错误

当前配置要求真实 Mamba-SSM。确认 `from mamba_ssm import Mamba2` 成功、项目导入的 `ProjectMamba2` 不为 `None`，并确认配置没有被改成其他主干；正式训练前执行 `python tools/verify_pipeline.py`。

## 官方参考

- [PyTorch 官方版本安装命令（2.13.0 / CUDA 13.2）](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 13.2 Linux 安装指南（含 Ubuntu 24.04）](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-installation-guide-linux/index.html)
- [CUDA 13.2 Toolkit Release Notes（驱动要求）](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
- [NVIDIA Driver / CUDA Toolkit 兼容矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/latest/supported-drivers-and-cuda-toolkit-versions.html)
- [Mamba 官方仓库与源码安装说明](https://github.com/state-spaces/mamba)
- [Mamba v2.3.2.post1 release（预编译 wheel 列表）](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1)
