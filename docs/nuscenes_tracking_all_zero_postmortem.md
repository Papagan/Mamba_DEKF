# nuScenes Tracking 全类别 0 分数问题复盘（防回归）

## 1. 现象（本次事故的识别特征）

当出现以下组合时，可判定为“评测有效执行，但跟踪完全失效”：

- `metrics_summary` 中所有类别：
  - `tp=0`
  - `recall=0`
  - `amota=0`
  - `fn` 接近 `gt`
- `results.json` 非空，但有效样本很少：
  - 例如：`samples=6019`，`nonempty_samples=443`
- 跟踪日志里关联全失败：
  - `stage1 matched=0`
  - `stage2 matched=0`
  - `status_1=0`（轨迹从未确认）
  - `output` 在前几帧后长期为 `0`
- 关联代价远超门限：
  - `[ASSOC] cost_min` 常见 `1e11~1e14`
  - `threshold` 仅 `3~9`
  - `below-threshold=0`

---

## 2. 根因（核心）

**时间戳单位不一致导致 `delta_t` 爆炸**：

- nuScenes 原始 `timestamp` 是微秒（us）。
- 推理/跟踪链路中，`tracker/base_tracker.py` 早期实现直接做 `timestamp - last_timestamp`，未统一到秒。
- 训练侧 `training/gt_tracklet_dataset.py` 对时间做过 `/1e6`，但推理侧未做同等处理。
- 结果：`delta_t` 被放大约 `1e6` 倍，KF 预测和协方差发散，不确定性代价失真，导致关联全面失败。

---

## 3. 触发链路（为何会突然“从有结果变成全 0”）

本次还叠加了配置自动切换影响：

- 配置文件 `config/nuscenes.yaml` 期望是 `THRESHOLD.BEV.COST_MODE: geometric`。
- 但运行日志出现 `BEV_COST_MODE=full`。
- 原因是训练流程 `training/train.py` 的 `AUTO_UNSEAL` 机制会在指定 epoch 后自动把 `nuscenes.yaml` 改为 `full`。
- 当 `full` 模式叠加错误 `delta_t` 时，问题被放大，最终所有类别无法匹配成功。

---

## 4. 已落地修复（当前代码）

### 4.1 统一 `delta_t` 单位（关键修复）

文件：`tracker/base_tracker.py`

- 新增 `delta_t` 归一化逻辑：
  - `>1e3` 视为微秒，除以 `1e6`
  - `>10` 视为毫秒，除以 `1e3`
  - 其余按秒处理
- 对 `delta_t` 做裁剪：`[1e-3, 5.0]` 秒。
- 角速度 `omega` 计算也复用同一 `dt` 归一化，避免旋转项异常。

### 4.2 full 代价分支异常保护（熔断回退）

文件：`tracker/matching.py`

- 在 `match_trajs_and_dets_uncertainty_aware` 的 `full` 模式里，若：
  - `feasible_pairs == 0`，或
  - 最小有限代价远超阈值（`min_cost > max_threshold * 50`），
- 则回退到 `geometric` 匹配，避免整段序列“全帧失配”。

---

## 5. 每次改程序前后的强制检查清单（防止再次踩坑）

### 改动前

- 确认 `config/nuscenes.yaml` 当前值是否被训练过程改写（重点看 `THRESHOLD.BEV.COST_MODE`）。
- 明确当前运行模式：`FILTER_MODE`、`BEV_COST_MODE`、`DATASET`。
- 开启调试日志（建议保留）：
  - `[TRK]` 帧级状态
  - `[ASSOC]` 代价统计
  - `[SAVE DIAG]` 结果落盘统计

### 运行中

- 采样检查前 `20~50` 帧：
  - `dt` 应接近数据真实帧间隔（nuScenes 常见约 `0.5s`），不应出现 `1e5+` 级别。
  - `stage1/stage2 matched` 必须出现非零。
  - `status_1` 应逐步增长（有确认轨迹）。
  - `output` 不应在 warm-up 后长期为 `0`。

### 评测前

- 检查 `results.json`：
  - `nonempty_samples / samples` 不应异常偏低。
  - `translation` 范围应在真实场景附近，避免离谱坐标。
- 如启用 `full` 模式，确认 `[ASSOC] cost_min` 与 `threshold` 同数量级，不应长期相差 `1e8+`。

### 评测后

- 若再次出现全类别 `tp=0`：
  1. 先查 `dt` 是否异常；
  2. 再查 `BEV_COST_MODE` 是否被自动改动；
  3. 再看 `stage1/stage2 matched` 是否持续为 0；
  4. 最后看输出过滤条件是否把轨迹全部压掉。

---

## 6. 建议保留的长期防护

- 保留 `delta_t` 归一化与裁剪，不要移除。
- 保留 `full -> geometric` 的熔断回退逻辑。
- 保留 `tools/clean_trk_assoc_log.py`，用于快速提取 `[TRK]` 和 `[ASSOC]` 定位关联失效。
- 若继续使用 `AUTO_UNSEAL`，务必在训练脚本和实验记录中明确写入“会改写推理配置”的警告。

---

## 7. 快速复现排查命令（参考）

```bash
python tools/clean_trk_assoc_log.py -i TRK.log -o filtered.log
```

重点检查 `filtered.log` 中：

- `dt=...s` 是否合理
- `stage1/stage2 matched` 是否非零
- `cost_min` 与 `threshold` 是否同量级

