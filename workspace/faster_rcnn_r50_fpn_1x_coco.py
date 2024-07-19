auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'mmdet.datasets.CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
    logger=dict(interval=50, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook'))
default_scope = None
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, type='mmengine.runner.LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='torch.nn.BatchNorm2d'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.models.backbones.resnet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type=
        'mmdet.models.data_preprocessors.data_preprocessor.DetDataPreprocessor'
    ),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.models.necks.fpn.FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type=
                'mmdet.models.task_modules.coders.delta_xywh_bbox_coder.DeltaXYWHBBoxCoder'
            ),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(
                loss_weight=1.0,
                type='mmdet.models.losses.smooth_l1_loss.L1Loss'),
            loss_cls=dict(
                loss_weight=1.0,
                type='mmdet.models.losses.cross_entropy_loss.CrossEntropyLoss',
                use_sigmoid=False),
            num_classes=80,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type=
            'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared2FCBBoxHead'
        ),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(
                output_size=7, sampling_ratio=0, type='mmcv.ops.RoIAlign'),
            type=
            'mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor.SingleRoIExtractor'
        ),
        type='mmdet.models.roi_heads.standard_roi_head.StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type=
            'mmdet.models.task_modules.prior_generators.anchor_generator.AnchorGenerator'
        ),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type=
            'mmdet.models.task_modules.coders.delta_xywh_bbox_coder.DeltaXYWHBBoxCoder'
        ),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            loss_weight=1.0, type='mmdet.models.losses.smooth_l1_loss.L1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            type='mmdet.models.losses.cross_entropy_loss.CrossEntropyLoss',
            use_sigmoid=True),
        type='mmdet.models.dense_heads.rpn_head.RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='mmcv.ops.nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='mmcv.ops.nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type=
                'mmdet.models.task_modules.assigners.max_iou_assigner.MaxIoUAssigner'
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type=
                'mmdet.models.task_modules.samplers.random_sampler.RandomSampler'
            )),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type=
                'mmdet.models.task_modules.assigners.max_iou_assigner.MaxIoUAssigner'
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type=
                'mmdet.models.task_modules.samplers.random_sampler.RandomSampler'
            )),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='mmcv.ops.nms'),
            nms_pre=2000)),
    type='mmdet.models.detectors.faster_rcnn.FasterRCNN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.02, momentum=0.9, type='torch.optim.sgd.SGD',
        weight_decay=0.0001),
    type='mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.001,
        type='mmengine.optim.scheduler.lr_scheduler.LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='mmengine.optim.scheduler.lr_scheduler.MultiStepLR'),
]
resume = False
test_cfg = dict(type='mmengine.runner.loops.TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='mmdet.datasets.transforms.Resize'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        test_mode=True,
        type='mmdet.datasets.CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='mmdet.evaluation.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type='mmdet.datasets.transforms.Resize'),
    dict(type='mmdet.datasets.transforms.LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.datasets.transforms.PackDetInputs'),
]
train_cfg = dict(
    max_epochs=12,
    type='mmengine.runner.loops.EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='mmdet.datasets.AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='mmdet.datasets.transforms.Resize'),
            dict(prob=0.5, type='mmdet.datasets.transforms.RandomFlip'),
            dict(type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        type='mmdet.datasets.CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.sampler.DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
    dict(type='mmdet.datasets.transforms.LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type='mmdet.datasets.transforms.Resize'),
    dict(prob=0.5, type='mmdet.datasets.transforms.RandomFlip'),
    dict(type='mmdet.datasets.transforms.PackDetInputs'),
]
val_cfg = dict(type='mmengine.runner.loops.ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='mmdet.datasets.transforms.Resize'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        test_mode=True,
        type='mmdet.datasets.CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='mmdet.evaluation.CocoMetric')
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.visualization.DetLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
work_dir = 'workspace'
