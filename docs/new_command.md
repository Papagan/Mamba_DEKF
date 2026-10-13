
## 当前数据生成、训练与验证命令

从仓库根目录执行。此技术路线冒烟测试按约定复用 val 数据；产出的验证指标不能作为泛化性能结论。

```bash
python tools/build_centerpoint_mini_train_dataset.py \
  --det-json /root/autodl-tmp/data/base_version/nuscenes/centerpoint/val.json \
  --nusc-dataroot /root/autodl-tmp/data/nuscenes/datasets/ \
  --nusc-version v1.0-trainval \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --max-scenes 150 \
  --dist-th 2.0 \
  --min-frames 2 \
  --min-matched-frames 2 \
  --scene-split val \
  --train-config config/train_nuscenes.yaml
python tools/augment_tracklet_cache_with_fusion.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --train-config config/train_nuscenes.yaml
cp /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
   /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl

python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --summary-output debug/train_pairwise_assoc_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0

python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/val_pairwise_assoc.pkl \
  --summary-output debug/val_pairwise_assoc_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0

python training/train.py --config config/train_nuscenes.yaml
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml -p 12
python tools/compare_pairwise_assoc_train_infer.py \
  --train-cache /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --infer-audit debug/association_head_audit.json \
  --output debug/association_shadow_decision_report.json \
  --min-known-changed 100
```






 重点是 association head 必须按类别拆分。bicycle/motorcycle/trailer 的关联逻辑和 car/pedestrian 明显不同，单头会被大样本类别主导，弱类很难得到最优边界。

  第一阶段：离线 pairwise association 训练
  当前 association_ranking_loss 只用 batch 内同类样本做 negative，这不够贴近真实匹配。真实错误通常发生在“同帧、同类、几何距离接近”的 hard negative 上。

  需要构造新的训练样本：

  (track_i, det_j, class_id, state_bucket, geometry_features, label)

  其中：

  - label=1：该 det 是 track 的真实后继。
  - label=0：同帧同类候选，但不是该 track。
  - hard negative：几何 cost 接近、距离近、分数高但 GT instance 不同。
  - easy negative：同类但距离远。
  - cross-class negative：只作为辅助稳定，不参与主匹配阈值学习。

  训练目标：

  L_assoc = BCE(pair_logit, label)
          + λ_rank * max(0, margin - logit_pos + logit_hard_neg)

  并且必须按类别监控：

  val/assoc/class_2_auc
  val/assoc/class_2_top1
  val/assoc/class_2_top3
  val/assoc/class_3_auc
  val/assoc/class_5_auc
  ...

  如果离线 top-1/top-3 不能超过纯几何 cost，这个 head 不允许接入主评估。

  第二阶段：多头关联 head 设计，不必局限于参数量
  每类一个轻量 head，不显著增加参数量：

  pair_feature = [
    track_embedding,
    det_embedding,
    abs(track_embedding - det_embedding),
    track_embedding * det_embedding,
    geometry_cost,
    center_distance,
    yaw_diff,
    size_diff,
    det_score,
    track_score,
    unmatch_length,
    state_bucket
  ]

  然后：

  assoc_logit = AssociationHead[class_id](pair_feature)

  class head 只负责本类候选，不跨类匹配。后续如果 class head 仍不够，可以升级为：

  AssociationHead[class_id][matched/unmatched]

  但第一版先做 7 类独立 head，避免过度复杂。

  第三阶段：推理接入方式，不能直接替代几何匹配。推荐 bounded correction：

  final_cost = geometric_cost + delta_cost
  delta_cost = clamp(alpha_class_state * penalty(logit), 0, max_delta)

  初期只允许 Mamba 做“惩罚不可信候选”，不允许强行救回几何上很差的候选。原因是历史实验已经证明，学习分支一旦覆盖几何主链，很容易出现 AMOTA 大幅下降。

  推荐接入策略：

  MAMBA_ASSOCIATION_PRIOR:
    ENABLED: false
    MODE: class_head_penalty
    MAX_DELTA: 0.03 ~ 0.08
    ACTIVE_CLASS_STATES:
      2: ["matched", "unmatched"]
      3: ["matched", "unmatched"]
      5: ["matched", "unmatched"]

  先只开弱类 A/B，再扩到全类。

  第四阶段：与状态头的关系
  状态头不删除，但定位要改变：

  - Association head：主提分路径。
  - State/noise head：辅助稳定路径。
  - Residual：只允许 bounded、小范围、按类按状态激活。
  - Q/R：继续 anchor 到当前 KF prior，不允许自由发散。

  也就是说：

  Mamba 不直接替代 KF；
  Mamba 学关联置信度 + 小幅不确定性修正；
  KF 继续负责稳定状态传播。

  第五阶段：评估门槛
  每一步都必须有回退条件：

  1. 离线 association 指标：
      - class top-1/top-3 必须优于 geometric baseline。
      - bicycle/motorcycle/trailer 至少一个类明显提升。

  2. nuScenes A/B：
      - 先 MAMBA_ASSOCIATION_PRIOR.ENABLED=false 验证新模型不破坏 0.740。
      - 再只开 class 2/3/5。
      - 最后再扩全类。

  3. 成功标准：
      - 总 AMOTA > 0.740。
      - 如果总分不升，即使某类提升，也不能合入主配置。
      - trailer/bicycle 的提升要看 FP/FN/MOTAR，不只看 AMOTA。

  我建议的实施顺序

  1. 新增 pairwise association cache/audit，先不训练。
  2. 统计每类候选数量、正负样本比例、hard negative 数量。
  3. 实现 ClassConditionedAssociationHead。
  4. 实现 pairwise association loss 和 per-class validation metrics。
  5. 训练共享 Mamba backbone + 多头 association。
  6. 评估时先关闭 association prior，确认状态链不掉分。
  7. 小幅打开 class 2/3/5 的 bounded association correction。
  8. 达到正收益后再考虑 matched/unmatched 二级 head。

  最终判断
  这条路线比继续调 residual 更值得做。它更贴近 AMOTA 的实际误差来源，也更能体现 Mamba 对时间上下文、检测语义、置信度和非线性运动的利用。关键是必须坚持：多头、按类监控、bounded 接入、默认可回退。


当前我们要的是构建一个TBD范式的mamba网络与滤波架构深度耦合的算法Mamba-DEKF，该算法利用检测算法输入的先验检测信息输入到多层Mamba主干网络提取时空间信息，然后分别学习到状态残差信息和数据关联信息，由于场景中的不同类别的目标在运动特性上存在显著差异，因此我们为每个类别设计了独立的状态残差学习模块和数据关联模块，此外我们还在轨迹管理阶段引入了一个基于类别的状态管理器，能够根据不同类别的运动特性和数据关联特性进行更精细的轨迹管理，从而提高多目标跟踪的准确性和鲁棒性。



• 是的，当前 association head 需要重新审计，问题不只是参数没调好，而是训练目标、网络输入、推理介入三者没有闭环一致。

  已确认的问题

  1. 类别 ID 映射是一致的，不是类别路由错误：
     car=0, pedestrian=1, bicycle=2, motorcycle=3, bus=4, trailer=5, truck=6。

  2. 训练和推理的 pair 几何特征不一致：
     训练时 center_distance/yaw_diff/size_l1 是 future positive det vs candidate det。
     推理时是 当前轨迹 bbox/预测 bbox vs candidate det。
     这意味着训练学的是 detection-detection 区分，评估却用在 track-detection 匹配上，score 标定会偏。

  3. 训练和推理的 history/embedding 输入不一致：
     训练 anchor_history 来自 cache 的 det/fusion feature。
     推理 trk_embeddings 来自主链路的 residual-token 或相对坐标 history。
     训练 candidate_history 是原始 obs feature，推理 detection history 是 [0,0,z,...] 这种相对形式。
     所以 best_assoc.pt 的 loss 可以很低，但评估时 embedding 分布已经变了。

  4. 当前介入方式过于粗糙：
     它只根据 score < MIN_SCORE 加 cost penalty，但这个 score 没有经过推理分布标定。结果就是它会惩罚一部分真实匹配，尤其 car/trailer 这类本来靠几何已经稳定的类，导致 AMOTA 从 0.740 掉到 0.738。

  我建议的重构方向
  推荐走“审计先行 + 轻量闭环重训”：
当前已经重新训练完成，下一步该执行下面哪一步了？
  1. 先增加 association score audit：
     统计每类 matched pair 的 score 分布、被 MIN_SCORE 惩罚的真匹配比例、惩罚后 Hungarian 选择是否改变。

  2. 重新构造训练样本：
     不再用 future_det vs candidate_det 作为 pair 几何，而是复刻推理阶段：
     track_state_after_predict/current track bbox vs candidate det。

  3. 统一 embedding 输入：
     pairwise association 训练必须调用和推理一致的 history 构造逻辑，尤其是 mamba_multihead_closure 下的 residual-token history，否则训练出来的 head 没有意义。

  4. 改介入方式：
     不直接全类加 penalty。先只作为“tie-breaker / margin reranker”或只在几何 cost 接近时介入。例如只对同类候选里 cost 差距小于 epsilon 的 pair 使用 association score，避免破坏明显正确的几何匹配。

  5. 评估开启策略：
     默认继续关闭 MAMBA_ASSOCIATION_HEAD，保住 0.740。只有 audit 显示某类真匹配惩罚率足够低，再按类开启。

  我建议下一步先实现 tools/audit_pairwise_association_inference.py，直接在评估结果或跟踪过程中导出每类 association score 的真实影响。没有这个审计，继续调 MIN_SCORE/ALPHA/MAX_DELTA 基本是在盲调。




  更具体一点：

  - score_pos_mean：正样本 sigmoid 后的平均分，应该从接近 0.5 逐步升高，理想状态到 0.8-0.95。如果很快到 0.999+，要警惕过拟合或任务太简单。
  - score_neg_mean：负样本平均分应该下降，理想到 0.05-0.2。重点是正负分数要拉开，而不是正样本单独升高。
  - top1：同一个 anchor 下，正样本是否排第一。应该逐步上升，越接近 1.0 越好。但 train top1 高、val top1 不高，就是过拟合。
  - hard_acc：正样本分数是否高于 hard negative。这个比普通 top1 更关键，应该持续上升。
  - logit_max：会随着正样本置信度升高而上升。合理范围通常是几到十几，例如 3-8 已经对应 sigmoid 0.95-0.9997。如果 logit_max > 20，基本说明分类器过度饱和。
  - logit_min：负样本越低越好，但如果低到 < -20，同样说明过度饱和。

  需要重点看 val，不要只看 train：

  健康：
  train score_pos ↑, score_neg ↓, top1 ↑
  val   score_pos ↑, score_neg ↓, top1 ↑

  过拟合：
  train top1 接近 1.0
  val top1 停滞或下降

  饱和：
  logit_max 很快 > 20
  logit_min 很快 < -20
  loss 接近 0
  val AMOTA/audit 没改善




    下一步最有效的不是调 ALPHA，而是做一个 train-vs-infer pair replay audit：

  1. 从一次推理中导出真实进入 matching 的 pair：
      - class
      - state bucket
      - cost_before
      - association score
      - 是否被 Hungarian 选中
      - track history embedding 输入摘要
      - det history embedding 输入摘要
      - pair geometry feature

  2. 从训练 cache 中抽同 scene/token/class 的 pair，比较：
      - pair geometry 是否一致
      - track history 是否一致
      - candidate history 是否一致
      - 正样本 score 是否显著高于推理中的 matched pair

  3. 如果训练内正样本 score 高、推理真匹配 score 低，就是输入分布错位。
  4. 如果训练内正样本 score 也低，那是训练/损失/模型本身失败。
  5. 如果训练和推理都能区分，但 AMOTA 不涨，才是介入方式或 ALPHA/MAX_DELTA 不足。



    推荐实施方案

  第一阶段只做审计闭环，不改变匹配结果：

  - 增加 shadow 模式，同时计算原始和关联修正后的 Hungarian 结果，但正式跟踪仍使用原始结果。
  - audit 写入配置哈希、checkpoint 哈希、git commit、有效类别状态表。
  - 记录每类/状态的 assignment changed、阈值跨越、原始与新选择、score gap。
  - 接入 GT instance 后统计“修正正确数、破坏正确数”，不再用生命周期 matched 代替 pair label。
  - 修复多进程 sample 合并。

  第二阶段重建真正的 inference-style cache：

  - 从实际 predict_before_associate() 后直接导出 track embedding、det embedding 和最终 pair vector。
  - 保存真实 RO_GDIOU_3D cost、candidate rank、cost gap、class/state 和 geometric gate。
  - 同时覆盖 matched、unmatched 和 no-match 样本。
  - 不再离线重构 fusion history，也不再用中心距离模拟线上 margin。

  第三阶段重训关联头：

  - 首轮冻结 Mamba 主干，只训练七类独立关联头。
  - 损失改为候选组级 listwise cross-entropy，加近邻 hard-negative ranking loss；BCE 只作为辅助。
  - 按 class/state 分桶采样和验证，保存每类最优 head。
  - 对每类/状态单独做 temperature calibration，取消统一 MIN_SCORE=0.60。

  第四阶段安全接入：

  - 只在几何前两名 cost gap 足够小时介入。
  - 使用候选间 score gap，而不是绝对 score 阈值。
  - 只允许单向惩罚，且 delta 仅覆盖当前几何差距，不使用固定全局增量。
  - car 首轮保持关闭；其他类别依据 shadow audit 的净正确修正率逐类启用。
  - 每个 class/state 单独 A/B，只有总 AMOTA 严格高于 0.740 才合并。

  下面的命令在项目根目录、存有 CenterPoint val.json 和 nuScenes 数据的机器上执行。它沿用你目前“训练与验证复用数据”的跑通方案，并重建修复后的缓存。

cd /home/alvin/demo/Mamba-DEKF
set -e

CACHE=/root/autodl-tmp/data/training_cache/nuscenes/mini
DET=/root/autodl-tmp/data/base_version/nuscenes/centerpoint/val.json
TRAIN_CFG=config/train_nuscenes.yaml
EVAL_CFG=config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml

# 1. CenterPoint 检测 + nuScenes GT → 检测轨迹缓存
python tools/build_centerpoint_mini_train_dataset.py \
  --det-json "$DET" \
  --nusc-dataroot /root/autodl-tmp/data/nuscenes/datasets/ \
  --nusc-version v1.0-trainval \
  --scene-split val \
  --max-scenes 0 \
  --use-default-class-dist-th \
  --min-frames 2 \
  --min-matched-frames 2 \
  --train-config "$TRAIN_CFG" \
  --inference-config "$EVAL_CFG" \
  --output "$CACHE/train_det.pkl"

# 2. 生成融合历史；当前跑通实验复用它作验证缓存
python tools/augment_tracklet_cache_with_fusion.py \
  --input "$CACHE/train_det.pkl" \
  --output "$CACHE/train_fusion.pkl" \
  --train-config "$TRAIN_CFG"
cp "$CACHE/train_fusion.pkl" "$CACHE/val_fusion.pkl"

# 3. 生成关联样本；当前跑通实验复用它作验证缓存
python tools/build_pairwise_association_cache.py \
  --input "$CACHE/train_fusion.pkl" \
  --output "$CACHE/train_pairwise_assoc.pkl" \
  --summary-output "$CACHE/train_pairwise_summary.json" \
  --train-config "$TRAIN_CFG" \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000
cp "$CACHE/train_pairwise_assoc.pkl" "$CACHE/val_pairwise_assoc.pkl"

# 4. 检查生成结果，然后训练
python tools/audit_det_tracklet_cache.py \
  --input "$CACHE/train_fusion.pkl" \
  --train-config "$TRAIN_CFG" \
  --output "$CACHE/train_fusion_audit.json"
python tools/audit_pairwise_association_cache.py \
  --input "$CACHE/train_pairwise_assoc.pkl" \
  --output "$CACHE/train_pairwise_audit.json"
python training/train.py --config "$TRAIN_CFG"

# 5. 在 nuScenes val 上运行跟踪与评估
python main.py --dataset nuscenes --eval --config "$EVAL_CFG" -p 12

训练配置 (config/train_nuscenes.yaml:128) 已设为主训练结束后自动训练关联头，无需再单独运行关联训练命令。检查 train_pairwise_summary.json 中的 unmatched_detection_negative_pairs，可确认新增负样本是否进入缓存。

当前评估配置 (config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml:137) 的 SHADOW_MODE: true，关联头只做旁路审计，不会改变最终匹配。若要验证关联头实际作用，复制该配置、在副本中设为 SHADOW_MODE: false，再用副本运行第 5 步。
