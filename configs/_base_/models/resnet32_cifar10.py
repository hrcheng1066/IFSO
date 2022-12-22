# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet32_CIFAR',
        depth=32,
        num_stages=3,
        out_indices=(2, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
