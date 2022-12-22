# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG16_CIFAR', depth=16, norm_cfg=dict(type='BN'), num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
