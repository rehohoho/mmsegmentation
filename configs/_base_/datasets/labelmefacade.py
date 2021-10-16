# dataset settings
PALETTE = [[0, 0, 0], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], 
  [128, 64, 0], [0, 128, 128], [0, 128, 0], [0, 0, 128]]
  
dataset_type = 'LabelmeFacadeDataset'
data_root = 'data/labelme_facade/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (192, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsFromPalette', palette=PALETTE),
    dict(type='Resize', img_scale=(512, 384), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
gt_pipeline = [
    dict(type='LoadAnnotationsFromPalette', palette=PALETTE),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 384),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline,
        gt_pipeline=gt_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='tests',
        ann_dir='labels',
        pipeline=test_pipeline))
