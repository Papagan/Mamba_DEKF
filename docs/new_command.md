
## 双 3090：数据生成、训练与验证命令

目录统一按 `/home/alvin/datasets/track_data/` 配置：`nuscenes/datasets/` 为 GT，
`base_version/nuscenes/centerpoint/` 为转换后的检测，
`training_cache/nuscenes/mini/` 为缓存，`checkpoints/mamba_motion/` 为权重和训练日志，
`results/nuscenes/` 为跟踪结果。这里仅修改代码路径，没有移动远程文件。
若检测、缓存或结果目录不是这个布局，先同步修改 train 和 eval 两份 YAML。

先激活远程已有的 Mamba2 环境。两张卡都应通过实际主干的前向和反向：

若提示 NumPy 1.x 编译扩展不能在 NumPy 2 下运行，先执行
`python tools/check_numpy_environment.py --output numpy_environment.json`，
处理报告中的失败扩展和依赖冲突，详见 [NumPy 兼容性说明](numpy_compatibility.md)。

```bash
cd /home/alvin/demo/Mamba-DEKF
python tools/verify_pipeline.py --device cuda:0 --require-cuda
python tools/verify_pipeline.py --device cuda:1 --require-cuda
```

没有 CUDA 时的 GRU 测试只验证接口，不能代替上述检查。

```bash
cd /home/alvin/demo/Mamba-DEKF
set -e

CACHE=/home/alvin/datasets/track_data/training_cache/nuscenes/mini
DET=/home/alvin/datasets/track_data/base_version/nuscenes/centerpoint/val.json
TRAIN_CFG=config/train_nuscenes.yaml
EVAL_CFG=config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml

# 1. CenterPoint 检测 + nuScenes GT → 检测轨迹缓存
python tools/build_centerpoint_mini_train_dataset.py \
  --det-json "$DET" \
  --nusc-dataroot /home/alvin/datasets/track_data/nuscenes/datasets/ \
  --nusc-version v1.0-trainval \
  --scene-split val \
  --max-scenes 0 \
  --use-default-class-dist-th \
  --min-frames 2 \
  --min-matched-frames 2 \
  --min-gt-sensor-points 1 \
  --min-identity-margin 0.15 \
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
  --inference-margin-metric runtime_geometric \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --allow-current-miss \
  --allow-future-miss \
  --sampling-seed 0 \
  --max-pairs-per-class car=80000,pedestrian=70000
cp "$CACHE/train_pairwise_assoc.pkl" "$CACHE/val_pairwise_assoc.pkl"

# 4. 检查生成结果，然后训练
python tools/audit_det_tracklet_cache.py \
  --input "$CACHE/train_fusion.pkl" \
  --train-config "$TRAIN_CFG" \
  --output "$CACHE/train_fusion_audit.json"
python tools/audit_pairwise_association_cache.py \
  --input "$CACHE/train_pairwise_assoc.pkl" \
  --train-config "$TRAIN_CFG" \
  --output "$CACHE/train_pairwise_audit.json"
python tools/audit_pairwise_association_cache.py \
  --input "$CACHE/val_pairwise_assoc.pkl" --train-config "$TRAIN_CFG"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 training/train.py --config "$TRAIN_CFG"

# 5. 必须用真实 tracker 历史完成关联头最终训练（需要带 audit_gt_tokens 的检测 JSON）
python tools/finalize_association_replay.py \
  --train-config "$TRAIN_CFG" --inference-config "$EVAL_CFG" \
  --cache-dir "$CACHE" --processes 4

# 6. 在 nuScenes val 上运行跟踪与评估
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset nuscenes --eval --config "$EVAL_CFG" -p 4
```

`BATCH_SIZE=128` 是全局 batch，双卡每卡 64；学习率不额外翻倍。
`NUM_WORKERS=4` 是每卡 worker 数，主机内存或 CPU 紧张时降为 2。
CUDA 数据加载进程使用 spawn，符合 [PyTorch 对 NCCL 多 worker 加载的要求](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)；它会复制数据集到 worker，主机内存紧张时可设为 0。
验证在 rank 0 上完整运行，只有 rank 0 写 checkpoint、TensorBoard 和启动后续关联训练。
分类关联头阶段使用一张卡，真实回放/评估通过 `INFERENCE_GPU_IDS:auto` 将场景 worker 分配到可见 GPU。
双卡训练使用各卡 batch 内的对比负样本，因而不保证与单卡 batch 128 数值等价。
当前保持 FP32 滤波与损失，不开启全局 FP16，以免影响矩阵求逆和协方差计算。
短序列、小模型的双卡通信成本可能抵消收益；以日志中的完整 epoch 时间比较，不能预设两倍提速。

断点续训使用完整的 `checkpoint_epochN.pt`，它包含 optimizer 和 scheduler；
`best.pt` 用于推理，不是完整续训快照：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 training/train.py \
  --config config/train_nuscenes.yaml \
  --resume /home/alvin/datasets/track_data/checkpoints/mamba_motion/checkpoint_epoch10.pt
```

单卡仍可用 `CUDA_VISIBLE_DEVICES=0 python training/train.py --config "$TRAIN_CFG"`。
同一输出目录不要同时启动两次训练。

训练配置已设为主训练结束后自动训练初始关联头，无需重复启动同一阶段；第 5 步的真实回放最终训练仍必须执行。检查 train_pairwise_summary.json 中的 unmatched_detection_negative_pairs，可确认新增负样本是否进入缓存。

当前主评估配置的 SHADOW_MODE: false，分类关联头参与最终匹配；Q/R 学习比例也已启用，与训练一致。旁路消融可在配置副本中改为 SHADOW_MODE: true。

本轮修复改变了融合状态和朝向协方差语义：旧 fusion 缓存和 motion checkpoint 不能直接复用，必须从第 2 步重新生成并训练；回放缓存升级为 schema v2，也必须重新导出。遇到状态契约错误不要绕过检查。

`TRACKER_COMPAT_MODE: mctrack` 启用 MCTrack 生命周期/匹配分支，改为 `default` 关闭。
两种情况下都保持 `STATE_OUTPUT_MODE: filter`，完整保留学习后的 z/yaw/尺寸/速度；切换兼容模式不会触发旧值覆盖。
已有的 `MCTRACK_EXACT_*` 滤波更新开关独立于生命周期分支，做兼容消融时保持与训练配置一致。
`legacy_mctrack` 状态写回只允许纯 DEKF 基线，学习分支请求该模式会明确报错。

`LEARNED_MOTION_MATCHING.ENABLED: true` 让所有类别的几何匹配直接使用本帧学习预测；
设为 false 才恢复下方原有的分类别 Predict/BackPredict/Fusion 表，不改变学习状态写回。
该开关与 `TRACKER_COMPAT_MODE` 独立：开启 MCTrack 生命周期不再隐式绕过学习预测。
`DATA.INCLUDE_SHORT_HISTORIES: true` 已覆盖启动/重生时的 1 帧输入，仍保留有效融合状态、身份和历史质量检查。
旧 checkpoint 没有该训练覆盖契约，需重新训练；真实检测框现在也传入了帧时间戳，与漏检框保持一致。
启动/重生时的初始角速度统一为 0，不从生命周期外借用差分角速度；融合缓存升级为 `fusion_feature_schema_version=2`，必须重新执行第 2、3 步后再训练。

## 漏检、候选和真实轨迹回放

生成器现在保留歧义检测和没有同类 GT 的帧；歧义身份不会强制标为负样本。
未达到最小帧数/匹配数的轨迹仍只进入候选池；合格轨迹的启动/重生短前缀现在参与运动训练。
`REQUIRE_CURRENT_MATCH=false` 只允许以仍存活的 fusion 状态结束的漏检窗口，
无有效检测/fusion 初始状态的窗口会被排除，不再使用 GT 初始化。
数据集启动时打印实际 `states={matched: ..., unmatched: ...}`；
关联审计同时输出 `state_buckets` 和 `negative_types`，可核对是否存在 unmatched 样本。
类别上限在随机排列完整 anchor 候选组后生效，不会截断一个组；实际对数可能略低于上限。

完成主干及离线初始关联训练后，必须用真实 tracker 历史再训练最终关联头。
BaseVersion 检测 JSON 需要带有 `audit_gt_tokens`：在原来的
`preprocess/convert_nuscenes.py` 转换命令末尾追加 `--audit_gt_enrichment`。
已有旧版身份标签也需要从原始检测重新转换；回放脚本要求新的 `audit_gt_policy.version=2`，防止沿用旧的歧义标签。
该字段只用于关联完成后的标签审计，不参与线上代价和匹配决策。
不要对已有的 BaseVersion JSON 再执行原始检测格式的转换。

```bash
python tools/finalize_association_replay.py \
  --train-config "$TRAIN_CFG" --inference-config "$EVAL_CFG" \
  --cache-dir "$CACHE" --processes 4
```

脚本只在临时的回放配置中允许初始权重，正式配置要求 `REQUIRE_TRAINING_SOURCE: tracker_replay`。
最终权重直接写入正式配置的 `MAMBA_ASSOCIATION_HEAD.CHECKPOINT_PATH`，默认 `/home/alvin/datasets/track_data/checkpoints/mamba_motion/best_assoc.pt`。
可以先加 `--dry-run` 只检查命令，不执行训练或推理。
有独立验证回放时，追加 `--val-replay /path/to/held_out_replay.pkl`；默认仍保留此前约定的跑通实验复用策略，不代表泛化验证。

回放默认每个类别/状态最多保留 256 组，每组最多 16 个候选，普通负候选限于最优几何代价 +0.10。
必留项包括几何最优、实际匹配、可信正样本（含重复身份以保留歧义）和一个未知身份标记；必留项超限时整组不导出，避免截断后制造错误标签。
组上限采用固定种子的哈希优先采样，跨 worker 合并结果与导出顺序无关，不偏向最早场景。
可在 `INFERENCE_PAIR_CACHE` 中调整 `MAX_GROUPS_PER_CLASS_STATE`、`MAX_CANDIDATES_PER_GROUP`、`CANDIDATE_COST_MARGIN`、`SAMPLING_SEED`。
这些限制只影响回放监督，不改变线上候选或最终匹配；同一路径的训练/验证复用缓存只加载、预计算一份。

上面继续沿用当前跑通实验的数据复用设置；正式泛化评估需要独立场景。
回放训练直接使用真实推理时的 pair vector，保留 ID 切换和连续漏检造成的历史变化。
训练器校验生成回放所用 motion checkpoint 的 SHA-256；换过主干后必须重新回放。
轨迹身份未知或全候选无可信标签时才排除整组；`partial` 组保留可信的正/负样本，仅屏蔽未知或重复身份候选。
可信的 no-match 组保留为全负 BCE 样本；回放标签与训练生成器复用同一套匹配/歧义判断逻辑，默认过滤零点 GT，身份 margin 为 0.15m。
先检查回放摘要中的有效 GT 组数；没有可信标签时训练会明确报错。
这一步覆盖关联训练的在线历史分布；运动预训练仍使用检测/GT 对齐轨迹，
运动分支对真实错误关联历史的鲁棒性需要在远程回放评估中验证。
