# Mamba-Decoupled-EKF (MDEKF) 3DMOT 深度融合架构重构技术文档

## 一、 核心架构升级定位 (Core Paradigm Shift)

本方案旨在重构当前项目中 `pure_ekf` 模式与 `mamba` 模式在协方差预测上面临的**数值尺度不统一、黑盒回归易发散、长尾弱类目标性能差劲**等核心痛点。

重构后的系统定位为 **物理启发的时序自适应倍率滤波框架 (Physics-Informed Temporal Adaptive Scale Filtering)**。其核心逻辑在于：**彻底放弃让 Mamba 神经网络直接预测协方差矩阵绝对值的做法**，转而将 Mamba 降级为“自适应动态调音师”——让 Mamba 提取多维运动耦合时序特征，联合类别嵌入（Category Embedding），输出相对于 EKF 静态物理先验噪声的**动态缩放因子（Scale Factors）**。

通过将绝对值回归转化为相对倍率控制，所有类别和状态量被强行拉平到统一的数学量纲中，彻底消除了梯度霸权，从而从底层数学逻辑上显著优化弱类目标的性能。

整个重构流水线严格遵循以下实体拓扑：


$$\text{历史残差输入} \longrightarrow \text{Mamba 时序特征提取} \longrightarrow \text{自适应任务分配 (Category Embedding)} \longrightarrow \text{专属任务头分支} \longrightarrow \text{可微 EKF 状态传播} \longrightarrow \text{抗噪后验损失约束}$$

---

## 二、 数据流与自适应多头重构 (Data Flow & Multi-Head Reconfiguration)

### 1. 输入层：新息脱水与空间对齐 (Input Standardization)

为了消除 CenterPoint 输出的绝对世界坐标带来的巨大数值洪峰，Mamba 网络的输入必须从“绝对状态”切换为“物理残差（新息空间）”。

* **残差提取**：在每一帧数据关联前，提取当前 CenterPoint 观测框 $z_t$ 与解耦卡尔曼滤波先验预测状态 $\hat{x}_{t|t-1}$ 的新息（Innovation）：

$$\tilde{y}_t = z_t - H \hat{x}_{t|t-1}$$


* **特征拼接**：将三维位置残差、尺寸残差、角度残差、检测置信度（Score）进行拼接。通过一个浅层全连接投影层对齐通道维度，并实施 `LayerNorm`，确保送入 Mamba 主干的网络输入始终保持“零均值、小方差”的平稳流形状态。

### 2. 动态任务分配与专属类别头 (Category-Conditioned Prior)

弱类目标（如行人、自行车）在训练集中样本稀少，其物理方差基数与主导类（如轿车）存在数量级差异，极易在联合训练中被“梯度饿死”。

* **网络实现**：在网络初始化中构建 `nn.Embedding(num_classes, class_embed_dim)`（如 `num_classes=10, embed_dim=16`）。
* **特征注入**：Mamba 主干网络提取出全局时序隐状态 $h_{mamba}$ 后，将当前轨迹的 `class_id` 转化为稠密嵌入向量，采用元素级相加（Element-wise Add）或特征拼接（Concat）的方式注入到隐状态中：

$$h_{task} = \text{Linear}(h_{mamba}) + \text{Embedding}(\text{class\_id})$$


* **学术收益**：这充当了自适应任务分配的“软路由钥匙”。网络不需要动态猜测去哪个专家分支，而是利用已知的类别先验作为条件门控（Condition），使后续的任务头能够自动感知目标的物理类别属性。

### 3. 专属任务头的物理锁设计 (Specialized Task Heads)

为位置、尺寸、朝向角三个完全解耦的物理维度，量身定制具有不同非线性物理边界的专用输出头，替代大而全的通用线性层。

* **物理约束架构 (ScaledTanhHead)**：
定义一个专门的输出头子模块，通过带边界的非线性映射卡死网络的输出范围。计算公式为：

$$\gamma = \exp\left( \alpha \cdot \tanh(W \cdot h_{task} + b) \right)$$


* **位置专属头（Position Head）**：
* 动态范围设定：$[0.1, 10.0]$（通过设定 $\alpha = \ln(10) \approx 2.3026$ 锁定范围）。
* 物理职责：当遇到极限加减速或大角度转弯等高动态运动时，迅速将位置过程噪声 $Q_{pos}$ 放大，增大对当前观测框的信任度。


* **尺寸专属头（Size Head）**：
* 动态范围设定：$[0.95, 1.05]$（极度缩紧的物理常数保护）。
* 物理职责：车辆作为刚体，其长宽高在物理上是常量。该头仅允许网络在 $\pm 5\%$ 范围内微调，彻底斩断由于网络权重扰动导致的尺寸高频剧烈形变。


* **朝向专属头（Orientation Head）**：
* 动态范围设定：$[0.5, 4.0]$。
* 物理职责：捕获车辆由于侧滑或传动系统极限机动引入的角度扰动。



---

## 三、 针对 CenterPoint 劣质数据的防御矩阵 (CenterPoint Noise Defense)

鉴于前端 CenterPoint 检测器存在“检测精度有限（带强噪声及假阳性）”、“训练样本少且分布极度不均（长尾问题）”的致命缺陷，系统不采纳时序运动学数据增强，完全通过以下三大底层防御机制硬抗噪声。

### 1. 基于检测得分的观测噪声 $R$ 动态释放 (Score-based R Scaling)

CenterPoint 输出的 `score` 包含了极其重要的置信度先验。当检测精度受限时，绝对不能让 EKF 的观测噪声矩阵 $R$ 保持静态常数，否则低质量检测框会直接拉偏并撕裂卡尔曼后验状态。

* **重构逻辑**：在送入 EKF 状态更新前，根据当前检测框的得分对基础观测噪声进行自适应硬放大：

$$R_{dynamic} = R_{base} \cdot \left(1.0 + \beta \cdot (1.0 - \text{score})\right)$$


* **物理意义**：当 CenterPoint 输出一个得分仅为 0.15 的弱目标时，$R_{dynamic}$ 会被瞬间放大数倍。EKF 滤波器会自动排斥该低质量观测，强制提升对物理预测模型的信任度，从而物理性地隔离了检测器的低级噪声，不让其污染 Mamba 的输入特征空间。

### 2. 鲁棒损失函数：降低离群点敏感度 (Outlier-Robust Loss)

传统的多元高斯负对数似然损失（Gaussian NLL）中包含残差的平方项，当 CenterPoint 产生严重的突发性误检（Outliers/假阳性）时，会产生极其恐怖的梯度洪峰，直接摧毁 Mamba 刚刚学到的时序收敛特征。

* **优化策略**：全面将位置和尺寸的损失函数从高斯 NLL 降级为 **Student-t 分布的负对数似然损失**，或在后验均值更新处引入 **Huber Loss (Smooth L1)** 算子。
* **数学效果**：在面对巨大离群误差时，Loss 的梯度会自适应趋于线性饱和，不再产生破坏性的指数级梯度破坏，实现网络对 CenterPoint 突发误检的天然免疫。

### 3. 长尾弱类的“逆频率重加权”与“物理枷锁收紧”

针对训练样本极少且分布不均的问题，实施双重保护策略：

* **逆频率重加权 (Inverse Frequency Weighting)**：统计训练集中各类别的轨迹样本总帧数，在计算最终的多任务联合 Loss 时，为少数弱势类别（如自行车、施工车辆）赋予与其出现频率成反比的动态损失权重，强迫网络在少数弱类样本出现时产生足够的梯度量级。
* **弱类物理枷锁收紧**：由于弱类数据量极少，Mamba 难以充分学到其完美的非线性时序流形。在代码初始化中，针对数据量稀少的长尾弱类，**主动将其任务头的允许缩放范围（即 $\alpha$ 值）缩减 50%**。
* **设计意图**：数据越少，越要相信经典控制论。收紧 Mamba 对弱类的发挥空间，逼迫系统退化并坚守在 `noise_priors.py` 中人工或审计标定的静态物理底盘上，确保基本盘不发散。

---

## 四、 滤波器矩阵维护与融合闭环 (Matrix Fusion Loop)

在 `base_tracker.py` 的运动学卡尔曼循环中，将专属任务头输出的动态缩放因子 $\gamma$ 以哈达玛乘积的形式乘入解耦卡尔曼滤波的过程噪声矩阵中：


$$Q_{fusion\_pos} = Q_{base\_pos} \odot \gamma_{pos}$$

$$Q_{fusion\_size} = Q_{base\_size} \odot \gamma_{size}$$

$$Q_{fusion\_ori} = Q_{base\_ori} \odot \gamma_{ori}$$

* **退化保护机制 (Fallback Guarantee)**：由于 $\gamma$ 是通过 $\exp(\cdot)$ 激活输出的，其值始终严格大于 0。这确保了缩放后的对角线协方差矩阵天然保持**严格的正半定性（PSD）**，彻底免除了由于矩阵奇异而导致的求逆崩溃。
* **物理退化状态**：当 Mamba 在训练初期或面对未知噪声由于拿不准而输出 0 时，$\gamma = \exp(0) = 1$，系统完美退化为纯正的、由物理专家调校的 `pure_ekf` 模式。这构建了整个深度融合框架最坚固的安全兜底边界。

---

## 五、 联合损失函数与训练调度策略 (Loss & Training Schedule)

由于 Mamba 输出的是相对缩放因子，没有 Ground Truth 标签，因此必须实施**端到端可微卡尔曼更新（Differentiable EKF Optimization）**。整个链条上的所有状态更新公式均采用 PyTorch 可导算子重写，Loss 最终作用于 EKF 更新后的**后验状态**上。

### 1. 朝向角定向拓扑损失 (Von Mises Loss)

朝向角存在特殊的 $[-\pi, \pi]$ 环形拓扑边界问题。若采用普通高斯损失，在边界截断处会产生虚假的断裂梯度。因此，朝向头引入定向统计学中的 **Von Mises 分布（环形正态分布）**，网络预测其集中度参数 $\kappa$。

* **贝塞尔防溢出工程重构**：为了避免第一类零阶修正贝塞尔函数 $I_0(\kappa)$ 在 $\kappa$ 变大时发生致命的浮点数上溢（Overflow），在 `losses.py` 中必须使用带指数缩放的 PyTorch 接口 `torch.special.i0e` 实施等价数学重构：

$$L_{ori} = \ln \text{i0e}(\kappa) + \kappa - \kappa \cdot \cos(\theta_{gt} - \theta_{post})$$



### 2. 训练调度：边界退火与锚点放缓策略 (Warm-up & Annealing)

为了防止 Mamba 在 Epoch 1 时其随机初始化的权重输出离谱的缩放因子直接干碎 EKF 的卡尔曼增益（Kalman Gain），必须在时间轴上配置严格的“物理教练导向策略”。

```
[Epoch 0-5]: 纯物理预热期  -> α限制在0.05 (近乎纯EKF), 强加λ_anchor阻尼
[Epoch 6-20]: 逐步放权退火期 -> α线性/余弦增长至最大值, λ_anchor指数级衰减
[Epoch 20+]: 完全自适应期   -> 解开全部枷锁, Mamba全力冲刺NLL/Huber误差下限

```

#### A. 尺度边界动态松绑（Alpha 退火）

不准直接在 Epoch 1 将 `ScaledTanhHead` 中的 $\alpha$ 设为最大边界值。

* **Epoch 0-5 (纯物理预热期)**：通过训练器调度，强制将 $\alpha$ 设置为极小值（如 $0.05$）。此时 $\gamma$ 被死锁在 $[0.95, 1.05]$ 之间。在这个阶段，网络作为“旁听生”，主要在学习如何稳定提取残差和类别特征，运动学状态完全由 EKF 物理底盘掌控。
* **Epoch 6-20 (逐步放权退火期)**：使用余弦退火（Cosine Annealing）调度器，将 $\alpha$ 从 $0.05$ 逐步平滑地放大到目标最大值（如 $\ln(10)$）。随着物理枷锁的慢慢解开，Mamba 获准开始微调协方差，拟合复杂的非线性机动。

#### B. 先验锚点损失放缓（Anchor Loss Relaxation）

为了在早期给予网络强大的恢复力阻尼，在联合损失函数中加入“先验锚点约束项”，强迫 Mamba 的输出 $\gamma$ 在初期尽量向 $1.0$ 靠拢：


$$L_{total} = L_{Student\_t\_NLL} + L_{VonMises} + \lambda_{anchor} \cdot \sum \|\ln(\gamma)\|$$

* **参数调度**：在 **Epoch 0-5**，设置极大的锚点惩罚权重 $\lambda_{anchor} = 10.0$，施加恐怖的强阻尼。进入 **Epoch 6** 后，启动指数衰减策略（Exponential Decay），让 $\lambda_{anchor}$ 随着 Epoch 稳步下降，在 Epoch 20 时彻底衰减至 $0.01$ 或 $0$，实现训练平稳软着陆，释放 Mamba 最终的非线性逼近潜力。

---

## 六、 关键工程踩坑指南与代码防爆建议 (Critical Caveats)

1. **物理转移矩阵的梯度阻断（Gradient Detach 锁）**
在通过 EKF 计算时序预测状态均值 $\hat{x}_{t|t-1} = F x_{t-1|t-1}$ 时，状态转移矩阵 $F$（包含物理时间步长 $\Delta t$）绝对不能参与梯度传导。**在代码实现中，必须确保对 $F$ 矩阵以及更新链路中涉及历史均值运算的部分执行 `.detach()` 操作**。梯度流必须仅仅通过协方差分支（$P, Q, R$ 矩阵）传回 Mamba 的预测头，否则网络会产生“逻辑妄想症”，试图去修改基础的微积分物理公式本身，导致物理底盘发生灾难性形变。
2. **推理阶段的 Coasting（盲推退化保护）**
在实际测试中，一旦当前轨迹由于 CenterPoint 长时间漏检而进入纯卡尔曼盲推（Coasting）状态，由于此时不再有新的新息残差产生，Mamba 的输入序列会发生严重的特征干涸。此时在 `base_tracker.py` 中，**必须强行切断 Mamba 的动态输出，将缩放因子 $\gamma$ 直接硬锁死为固定的 $1.0$**。通过全面退化为静态物理外推，保证系统在极端未知盲区下的基础鲁棒性。