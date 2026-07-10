

python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml -p 12
  python tools/merge_class_heads.py \
    --base checkpoints/mamba_dekf/best.pt \
    --class-head 0=checkpoints/mamba_dekf/best_class_0.pt \
    --class-head 1=checkpoints/mamba_dekf/best_class_1.pt \
    --class-head 2=checkpoints/mamba_dekf/best_class_2.pt \
    --class-head 3=checkpoints/mamba_dekf/best_class_3.pt \
    --class-head 4=checkpoints/mamba_dekf/best_class_4.pt \
    --class-head 5=checkpoints/mamba_dekf/best_class_5.pt \
    --class-head 6=checkpoints/mamba_dekf/best_class_6.pt \
    --output checkpoints/mamba_dekf/best_merged_classes_3_5.pt

python - <<'PY'
import torch
ckpt = torch.load("checkpoints/mamba_dekf/best.pt", map_location="cpu")
sd = ckpt.get("model_state_dict", ckpt)
print(ckpt.get("runtime_contract", {}))
print("fallback_gru keys:", sum(k.startswith("fallback_gru") for k in sd))
print("mamba_layers keys:", sum(k.startswith("mamba_layers") for k in sd))
PY



python tools/build_centerpoint_mini_train_dataset.py \
  --det-json /root/autodl-tmp/data/base_version/nuscenes/centerpoint/val.json \
  --nusc-dataroot /root/autodl-tmp/data/nuscenes/datasets/ \
  --nusc-version v1.0-trainval \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --max-scenes 150 \
  --dist-th 2.0 \
  --min-frames 2 \
  --min-matched-frames 2 \
  --train-config config/train_nuscenes.yaml
python tools/augment_tracklet_cache_with_fusion.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --train-config config/train_nuscenes.yaml
cp /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl /root/autodl-tmp/data/training_cache/nuscenes/train_fusion.pkl /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl 

  python tools/build_pairwise_association_cache.py \
    --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
    --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
    --summary-output docs/train_pairwise_assoc_summary.json \
    --train-config config/train_nuscenes.yaml \
    --history-len 8 \
    --future-step 1 \
    --hard-negative-distance 4.0 \
    --max-hard-negatives 4 \
    --max-easy-negatives 2
  
  python tools/build_pairwise_association_cache.py \
    --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
    --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
    --summary-output docs/train_pairwise_assoc_summary.json \
    --train-config config/train_nuscenes.yaml \
    --max-hard-negatives 4 \
    --max-easy-negatives 2 \
    --max-pairs-per-class car=80000,pedestrian=70000

  python tools/build_pairwise_association_cache.py \
    --input /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl \
    --output /root/autodl-tmp/data/training_cache/nuscenes/mini/val_pairwise_assoc.pkl \
    --summary-output docs/val_pairwise_assoc_summary.json \
    --train-config config/train_nuscenes.yaml

  python tools/audit_pairwise_association_cache.py \
    --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
    --output docs/train_pairwise_assoc_audit.json

  python tools/build_pairwise_association_cache.py \
    --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
    --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
    --summary-output docs/train_pairwise_assoc_summary.json \
    --train-config config/train_nuscenes.yaml


  python tools/train_pairwise_association.py --train-config config/train_nuscenes.yaml



python main.py --dataset nuscenes --eval  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml     -p 12



为什么执行了python training/train.py --config config/train_nuscenes.yaml后checkpoint中保存的模型并没有best_assoc.pt


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