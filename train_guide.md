nuScenes 的 v1.0-trainval 需要:
1. Metadata — v1.0-trainval_meta.tgz（必须，包含所有标注、instance_token、sample_annotation）
因为我们的训练管线只用 GT 标注数据（位置、速度、尺寸、yaw、track ID），这些全在 Metadata 里。不需要点云或图像。只下载 Metadata 就够了：
    Full dataset (v1.0)
        └── Metadata  ← 只需要这个（~1.4GB）
下载后解压到：
    data/nuscenes/datasets/ 
     └── v1.0-trainval/ 
        ├── attribute.json
        ├── category.json
        ├── instance.json
        ├── ...
# 下载 v1.0-mini，解压到 data/nuscenes/datasets/
然后改配置：
    DATA:  
        NUSC_VERSION: v1.0-mini 
        NUSC_DATAROOT: data/nuscenes/datasets/
        TRAIN_SPLIT: mini_train
        VAL_SPLIT: mini_val
