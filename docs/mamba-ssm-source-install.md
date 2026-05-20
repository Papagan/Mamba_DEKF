# mamba-ssm 源码安装指南

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 (V12.1.105) |
| GCC | 14.3.0 |

## 操作步骤

```bash
# 第1步：安装编译工具链
#  ————————————————————————————————————————————————
pip install ninja packaging wheel setuptools

# 确认 nvcc 可用
nvcc --version

# 第2步：卸载旧 wheel（如果装了）
#  ————————————————————————————————————————————————
pip uninstall mamba-ssm selective-scan -y

# 第3步：克隆官方仓库（v2.2.x 系列兼容 PyTorch 2.1.x）
#  ————————————————————————————————————————————————
cd /tmp
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.4   # 稳定 release，不要用 main

# 第4步：源码编译安装
#  ————————————————————————————————————————————————
# --no-build-isolation  → 用当前环境的 PyTorch 头文件编译 CUDA 扩展
# --no-cache-dir         → 强制重新编译，不走缓存
pip install . --no-build-isolation --no-cache-dir -v

# -v 会打印完整编译日志。如果中途报错，保留日志方便排查。

# 第5步：验证
#  ————————————————————————————————————————————————
python -c "from mamba_ssm import Mamba; print('OK')"
```

---

## 常见编译问题

### 1. `fatal error: cuda_runtime.h: No such file or directory`

```bash
# 确认 CUDA 头文件路径
ls /usr/local/cuda/include/cuda_runtime.h

# 如果不存在，设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda
# 或者在 conda 环境里
export CUDA_HOME=$CONDA_PREFIX
```

### 2. `ninja: build stopped: subcommand failed` — 编译器不兼容

GCC 14.3 和 CUDA 12.1 有已知兼容问题。降级到 GCC 12：

```bash
conda install -c conda-forge gcc=12 gxx=12
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/g++
```

### 3. `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`

编译时没有报错但 import 时报错 → PyTorch ABI 不匹配。确认：

```bash
# 源码编译时是否用的是同一个 PyTorch
python -c "import torch; print(torch.__file__)"
# 清除残留 .so 缓存
find . -name "*.so" -path "*/mamba_ssm/*" -delete
pip install . --no-build-isolation --no-cache-dir -v --force-reinstall
```

### 4. 内存不足导致编译失败

```bash
# 限制并行编译线程
export MAX_JOBS=4
pip install . --no-build-isolation --no-cache-dir -v
```

---

## 如果还是不行：使用 v2.2.2 更保守

```bash
cd /tmp/mamba
git checkout v2.2.2
pip install . --no-build-isolation --no-cache-dir -v
```

v2.2.2 对 PyTorch 2.1 的兼容性更好，CUDA 扩展接口更稳定。

---

## 验证安装完整性

```bash
# 注意：用 submodule 直接导入，避免 mamba_ssm.__init__.py 触发
# transformers 依赖链（项目中已改用此方式）
python -c "
import torch
from mamba_ssm.modules.mamba_simple import Mamba
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 4, 64).cuda()
y = model(x)
print(f'Input: {x.shape}, Output: {y.shape}')
print('Mamba forward OK')
for n,_ in model.named_parameters():
    if 'A_log' in n or 'dt_proj' in n:
        print(f'Found: {n}'); break
else:
    print('WARNING: no Mamba params')
"
```

---

## 备注

- 如果当前训练跑的是 GRU fallback，用 Mamba 从头训练需要**重新调 MAMBA_LR**（Mamba 学习率通常比 GRU 低 3–5 倍）
- Mamba 推理速度通常比 GRU 慢 20–30%（多了 selective scan），但你的 T=4 序列太短，差异可忽略
- 源码安装保留了 GRU fallback 代码路径——如果 `Mamba is None` 自动回退，不影响已有检查点加载
