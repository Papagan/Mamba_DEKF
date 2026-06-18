# Project Context Log

本文档是本项目的长期上下文日志。后续在修改训练、匹配、滤波、配置前，先快速检查本文件，避免重复踩已经确认过的坑。

适用范围：
- `tracker/`
- `training/`
- `kalmanfilter/`
- `dataset/`
- `config/nuscenes.yaml`
- `config/train_nuscenes.yaml`
- `tools/build_centerpoint_mini_train_dataset.py`

---

## 1. 项目主链路

### 1.1 推理/评估主链

入口在 [main.py](/home/alvin/demo/Mamba-DEKF/main.py)：

1. 读取 `config/*.yaml`
2. 从 `DETECTIONS_ROOT/DETECTOR/SPLIT.json` 读取 BaseVersion 检测结果
3. 用 [dataset/baseversion_dataset.py](/home/alvin/demo/Mamba-DEKF/dataset/baseversion_dataset.py) 做：
   - 类别过滤
   - `INPUT_SCORE` 过滤
   - nuScenes NMS
4. 每帧送入 [tracker/base_tracker.py](/home/alvin/demo/Mamba-DEKF/tracker/base_tracker.py)：
   - `predict_before_associate()`
   - 匹配
   - KF update
   - 生命周期管理
   - `get_output_trajs()`
5. 最后写出 `results.json` 并调用评测

### 1.2 跟踪核心

[tracker/base_tracker.py](/home/alvin/demo/Mamba-DEKF/tracker/base_tracker.py) 是整个系统最关键的文件：

- `Base3DTracker.track_single_frame()`
  - 计算 `delta_t`
  - 构造轨迹历史
  - 调 `MambaDecoupledEKF.predict_with_mamba`
  - 匹配
  - 更新匹配轨迹
  - 未匹配轨迹 `unmatch_update`
  - birth / death
- `get_output_trajs()`
  - 最终输出分数不是原始检测分数，而是轨迹内部聚合后的 `bbox.det_score`

### 1.3 匹配核心

[tracker/matching.py](/home/alvin/demo/Mamba-DEKF/tracker/matching.py) 负责：

- `match_trajs_and_dets()`
  - 纯几何/RV 匹配
- `match_trajs_and_dets_uncertainty_aware()`
  - `full` 模式下融合几何、语义 embedding、不确定性项

[tracker/cost_function.py](/home/alvin/demo/Mamba-DEKF/tracker/cost_function.py) 决定具体代价。

### 1.4 生命周期核心

[tracker/trajectory.py](/home/alvin/demo/Mamba-DEKF/tracker/trajectory.py) 只负责：

- `track_length`
- `status_flag`
- `matched_scores`
- `unmatch_length`
- 轨迹历史缓存

它**不再自己做 KF**。KF 全在 `kalmanfilter/` 和 `base_tracker.py`。

---

## 2. 训练主链路

入口在 [training/train.py](/home/alvin/demo/Mamba-DEKF/training/train.py)。

当前训练有两种数据源：

- `TRAIN_SOURCE=gt`
  - 使用 [training/gt_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/gt_tracklet_dataset.py)
- `TRAIN_SOURCE=det`
  - 使用 [training/det_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/det_tracklet_dataset.py)

当前更重要的是 `det` 模式，因为它更接近推理输入分布。

训练主流程：

1. 读取 tracklet cache
2. 构造 `track_history`
3. `TemporalMamba` 一次前向预测：
   - `Q_pos / Q_siz / Q_ori`
   - `R_pos / R_siz / R_ori`
   - `kappa_ori`
   - `embedding`
   - `delta_pos`
4. 用 `DecoupledAdaptiveKF` 做 rollout
5. 用 `JointLoss` 做监督

---

## 3. 当前训练/推理契约

### 3.1 历史特征定义必须一致

训练和推理当前都依赖 12 维特征：

`[dx, dy, z, vx, vy, vz, l, w, h, yaw, omega, det_score]`

这个定义在以下两边必须保持一致：

- 训练侧：
  - [training/det_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/det_tracklet_dataset.py)
  - [training/gt_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/gt_tracklet_dataset.py)
- 推理侧：
  - [tracker/base_tracker.py](/home/alvin/demo/Mamba-DEKF/tracker/base_tracker.py)

### 3.2 时间戳必须统一到秒

这是本项目最危险的坑之一。

历史结论：
- nuScenes 原始 `timestamp` 是微秒
- 训练 cache 必须用秒
- 推理阶段 `delta_t` 也必须归一化到秒

当前已经修过：
- 训练 cache 构造时，用 nuScenes 官方 `sample["timestamp"] / 1e6`
- 推理侧在 [tracker/base_tracker.py](/home/alvin/demo/Mamba-DEKF/tracker/base_tracker.py) 做了 `delta_t` 单位自适应归一

如果未来出现：
- 全类别 `tp=0`
- `cost_min` 极大
- 前几帧后 `output=0`

先查时间戳和 `delta_t`，不要先调阈值。

参考复盘：
- [docs/nuscenes_tracking_all_zero_postmortem.md](/home/alvin/demo/Mamba-DEKF/docs/nuscenes_tracking_all_zero_postmortem.md)

### 3.3 miss 和 padding 不能混淆

当前 detection-driven 训练里已经区分：

- `history_mask`
- `history_match_mask`

条件噪声在 [kalmanfilter/noise_priors.py](/home/alvin/demo/Mamba-DEKF/kalmanfilter/noise_priors.py) 中使用：

- `valid_ratio`
- `matched_ratio`
- `miss_ratio`

以后如果改 dataset 或 history 构造，不能再退回“全零即 miss/pad 通吃”的旧逻辑。

### 3.4 `mctrack_compat` 重训契约

当前已经确认：

- `TRACKER_COMPAT_MODE=mctrack, FILTER_MODE=pure_dekf`
  明显优于
- `TRACKER_COMPAT_MODE=default, FILTER_MODE=pure_dekf`

因此后续如果要继续训练 `mamba / fusion`，训练输入必须和 `mctrack` 推理壳子的历史语义一致。

当前正确链路应为：

1. 先用 [tools/build_centerpoint_mini_train_dataset.py](/home/alvin/demo/Mamba-DEKF/tools/build_centerpoint_mini_train_dataset.py) 从 `centerpoint/val.json` 生成 `train.pkl`
2. 再用 [tools/augment_tracklet_cache_with_fusion.py](/home/alvin/demo/Mamba-DEKF/tools/augment_tracklet_cache_with_fusion.py) 生成 `train_fusion.pkl`
3. 训练配置中使用：
   - `HISTORY_SOURCE: fusion`
   - `INIT_STATE_SOURCE: fusion`
   - `TRAIN_TRACKER_COMPAT_MODE: mctrack`
   - `EXPECTED_BEV_COST_MODE: geometric`
4. 推理配置中使用：
   - `TRACKER_COMPAT_MODE: mctrack`

如果 `DetectionTrackletDataset` 配置成 `fusion`，但 cache 仍然是旧的 `train.pkl`，现在会直接报错，不允许静默退回错误语义。

---

## 4. 当前技术路线的真实含义

当前项目**不是端到端直接输出 tracking id 的网络**。

当前路线是：

1. `TemporalMamba` 学：
   - 时序特征
   - 噪声残差
   - embedding
   - 当前状态残差
2. `DecoupledAdaptiveKF` 负责：
   - 位置
   - 尺寸
   - 朝向
   的独立滤波
3. `tracker/base_tracker.py` 负责：
   - 匹配
   - birth
   - confirm
   - coast
   - death

因此：

- 训练 loss 正常下降
- 不代表最终 tracking 指标必然好

很多最终指标仍由匹配策略和生命周期规则决定。

---

## 5. 当前已经确认过的关键问题

### 5.1 “评测全 0”不是评测坏了，通常是匹配链断了

已确认过的根因：

- `delta_t` 单位错误
- `COST_MODE` 被意外切到 `full`
- 关联代价整体爆炸

排查入口：

- `[TRK]`
- `[ASSOC]`
- `[SAVE DIAG]`

### 5.2 `full` 模式目前不一定优于 `geometric`

这不是偶然。

原因已确认：

1. `full` 的代价量纲和 `geometric` 不同
2. detection embedding 仍然比 trajectory embedding 更不稳定
3. uncertainty 项会抬高整条轨迹行的代价
4. 直接复用 geometric 模式调好的 `COST_THRE` 往往会变差

结论：

- `full` 模式必须单独重调
- 不要把 `geometric` 的阈值原样搬过去

### 5.3 小类问题常常不是“完全匹配不到”，而是“匹配后难确认”

`bicycle` / `motorcycle` 已经确认过：

- 之前有过 `matched_score` 被错误传成 `0.0`
- 之前有过 `track_length > confirmed_len` 的 off-by-one
- 之前有过 `BackPredict/Fusion` + 检测速度不稳 导致几何代价差

当前已经修过：

- 真实关联代价传入 `Trajectory.update()`
- 确认逻辑改成“累计 5 帧就确认”

但小类依然可能卡在：

- `CONFIRMED_DET_SCORE`
- `CONFIRMED_MATCHED_SCORE`
- `COST_STATE`
- `OUTPUT_SCORE`

### 5.4 单阶段比 ByteTrack 强，主因通常是 birth 策略，不是 stage2 匹配坏了

这在当前项目里已经确认过。

原因：

- 单阶段：所有未匹配检测都能 birth
- `ByteTrack`：原本只有 `high_dets` 能 birth
- 当前 CenterPoint 的分数标定偏保守，很多真阳性在 `INPUT_SCORE < score < BIRTH_SCORE`

当前已经补了：

- [tracker/bytetrack_utils.py](/home/alvin/demo/Mamba-DEKF/tracker/bytetrack_utils.py)
- `TENTATIVE_BIRTH_SCORE`

现在 ByteTrack 支持：

- `high`：严格 birth
- `tentative`：可建临时轨
- `low`：只救活不建轨

### 5.5 `BIRTH_SCORE` 在单阶段下基本不重要

当 `USE_BYTETRACK=False` 时：

- 单阶段里所有未匹配检测都 birth
- 真正重要的是：
  - `INPUT_SCORE`
  - `COST_THRE`
  - `CONFIRMED_TRACK_LENGTH`
  - `CONFIRMED_DET_SCORE`
  - `CONFIRMED_MATCHED_SCORE`
  - `OUTPUT_SCORE`

不要把 `ByteTrack` 的 birth 经验直接套到单阶段。

---

## 6. detection-driven 训练的当前状态

### 6.1 已完成的部分

当前 detection-driven 训练已经具备：

- detection history
- detection current state
- detection future observation
- GT future supervision
- history mask / match mask
- conditional noise
- residual anchor
- adaptive windows
- per-class adaptive windows

### 6.2 训练 cache 的来源与风险

当前用于流程验证的 cache 往往来自：

- `centerpoint/val.json`

不是正式 train 检测。

这意味着：

- 可以验证训练链是否跑通
- 不能代表最终训练上限

当前关键工具：

- [tools/build_centerpoint_mini_train_dataset.py](/home/alvin/demo/Mamba-DEKF/tools/build_centerpoint_mini_train_dataset.py)
- [tools/audit_det_tracklet_cache.py](/home/alvin/demo/Mamba-DEKF/tools/audit_det_tracklet_cache.py)

### 6.3 样本审计的结论不要忘

之前已经确认过：

- 小类 `bicycle/motorcycle` 样本量少
- 固定 `HISTORY_LEN + ROLLOUT_STEPS` 会丢很多短轨迹
- 训练 cache 中小类真阳性分数均值往往低于推理 `birth/confirm` 阈值

所以后来加了：

- adaptive windows
- per-class adaptive windows
- 更保守的 uncertainty warmup

如果后续再看到“小类训练了也没提升”，先回头看 cache 审计结果，不要先怀疑 Mamba 主干。

---

## 7. 关键配置的解释

### 7.1 `config/nuscenes.yaml`

最关键的是这几组：

- `THRESHOLD.INPUT_SCORE`
  - 输入检测过滤
- `THRESHOLD.BEV.COST_THRE`
  - 匹配门限
- `THRESHOLD.TRAJECTORY_THRE.*`
  - 生命周期门控
- `MATCHING.USE_BYTETRACK`
  - 单阶段 / 两阶段
- `MATCHING.BEV.COST_STATE`
  - `Predict / BackPredict / Fusion`
- `DEKF_BASE_NOISE`
  - 推理时的滤波先验

### 7.2 `config/train_nuscenes.yaml`

当前重点看：

- `TRAIN_SOURCE / VAL_SOURCE`
- `TRAIN_TRACKLET_PKL / VAL_TRACKLET_PKL`
- `MODEL.HISTORY_LEN`
- `TRAINING.ROLLOUT_STEPS`
- `TRAIN_ADAPTIVE_WINDOWS`
- `CLASS_WINDOW`
- `BASE_NOISE`
- `WARMUP_UNCERTAINTY_EPOCHS`
- `WARMUP_TRANSITION_EPOCHS`

### 7.3 训练和推理噪声先验现在已经对齐

之前训练 `BASE_NOISE` 和推理 `DEKF_BASE_NOISE` 不完全一致。

当前已经对齐了：

- `Q.POS_PER_CAT`
- `R` 统计
- `CONDITIONAL_NOISE`
- `RESIDUAL_ANCHOR.TARGETS` 包含 `Q_pos`

以后如果只改一边，记得同时检查另一边。

---

## 8. 当前建议的调参顺序

### 8.1 如果单阶段结果不好

先调：

1. `INPUT_SCORE`
2. `COST_THRE`
3. `CONFIRMED_TRACK_LENGTH`
4. `CONFIRMED_DET_SCORE`
5. `CONFIRMED_MATCHED_SCORE`
6. `OUTPUT_SCORE`

不要先去调 `BIRTH_SCORE`。

### 8.2 如果 ByteTrack 结果不好

先调：

1. `BIRTH_SCORE`
2. `TENTATIVE_BIRTH_SCORE`
3. `CONFIRMED_DET_SCORE`
4. `CONFIRMED_TRACK_LENGTH`
5. `stage2 relaxed COST_THRE` 相关逻辑

### 8.3 如果 full 模式结果不好

先调：

1. `W_UNCERTAINTY`
2. `W_SEMANTIC`
3. `COST_THRE`

不要先怀疑评测器。

### 8.4 如果训练 loss 正常但 tracking 不提升

优先排查：

1. 训练 cache 样本质量
2. 训练/推理阈值不一致
3. 生命周期门控
4. 小类样本太少
5. 匹配策略本身

不要第一反应就去堆 epoch。

---

## 9. 当前建议保留的调试入口

### 9.1 环境变量日志

常用：

- `DEBUG_ASSOC=1`
- `DEBUG_TRACKER=1`
- `DEBUG_SMALL_CLASSES=1`
- `DEBUG_KEY_LOG=...`

### 9.2 关键日志前缀

- `[ASSOC]`
- `[TRK]`
- `[TRK-SMALL]`
- `[SAVE DIAG]`

### 9.3 已有清洗工具

- [tools/clean_trk_assoc_log.py](/home/alvin/demo/Mamba-DEKF/tools/clean_trk_assoc_log.py)
- [tools/clean_result_log.py](/home/alvin/demo/Mamba-DEKF/tools/clean_result_log.py)
- [tools/clean_train_log.py](/home/alvin/demo/Mamba-DEKF/tools/clean_train_log.py)

---

## 10. 修改前的最小自检

在忘记细节、或准备改动主链路时，先回答这几个问题：

1. 当前跑的是：
   - `geometric` 还是 `full`
   - `USE_BYTETRACK=True` 还是 `False`
   - `gt` 训练还是 `det` 训练
2. 当前问题发生在：
   - 输入过滤
   - 匹配
   - 确认
   - 输出分数
   - 评测
3. 当前异常更像：
   - 召回不足
   - FP 太多
   - 匹配全失败
   - 小类确认不上
   - 训练/推理分布错位

如果这 3 个问题都还没回答清楚，不要直接改代码。

---

## 11. 本文档的维护原则

以后只有两类信息才应该追加到这里：

1. 已经反复验证过、容易再次遗忘的项目事实
2. 已经定位过根因、且后续改动很容易再次踩中的坑

不要把一次性的实验结果、临时命令输出、大段日志原样贴进来。那类内容应放在：

- `docs/result.log`
- `docs/noise.log`
- 单独的 postmortem 文档

---

## 12. 必读关联文档

- [docs/nuscenes_tracking_all_zero_postmortem.md](/home/alvin/demo/Mamba-DEKF/docs/nuscenes_tracking_all_zero_postmortem.md)
- [docs/result.log](/home/alvin/demo/Mamba-DEKF/docs/result.log)
- [docs/noise.log](/home/alvin/demo/Mamba-DEKF/docs/noise.log)
- [README.md](/home/alvin/demo/Mamba-DEKF/README.md)

## 2026-06-17: Frozen Exact-Hybrid Baseline

- Baseline config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Baseline aggregated AMOTA: `0.737`
- Baseline purpose: immutable A/B reference for all route-A work
- Constraint: any new change must be evaluated against this config and must not replace it in-place
