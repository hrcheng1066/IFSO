_base_ = [
    '../../_base_/models/resnet32_cifar10.py', '../../_base_/datasets/cifar10_bs128.py',
    '../../_base_/schedules/cifar10_bs128_2.py', '../../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=10))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

custom_hooks = [
    dict(
        type='FisherPruningHook',
        pruning=True,
        batch_size=128,
        interval=10,
        priority='LOWEST',
    )
]

load_from = '../result/01-resnet32-cifar10-fullnetwork-channel-16/epoch_181_93.51.pth'