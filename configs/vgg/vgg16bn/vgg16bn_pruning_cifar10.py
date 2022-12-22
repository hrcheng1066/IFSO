_base_ = [
    '../../_base_/models/vgg16bn_cifar10.py', '../../_base_/datasets/cifar10_bs128.py',
    '../../_base_/schedules/cifar10_bs128_2.py', '../../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

custom_hooks = [
    dict(
        type='FisherPruningHook',
        # In pruning process, you need set priority
        # as 'LOWEST' to insure the pruning_hook is excused
        # after optimizer_hook, in fintune process, you
        # should set it as 'HIGHEST' to insure it excused
        # before checkpoint_hook
        pruning=True,
        batch_size=128,
        interval=10,
        priority='LOWEST',
    )
]

load_from = '../result/01-vgg16bn-cifar10-fullnetwork/epoch_198_93.34.pth'