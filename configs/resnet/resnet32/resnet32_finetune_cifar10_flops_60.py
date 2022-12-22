_base_ = [
    '../../_base_/models/resnet32_cifar10.py', '../../_base_/datasets/cifar10_bs128.py',
    '../../_base_/schedules/cifar10_bs128_2.py', '../../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='FisherPruningHook',
         pruning=False,
         deploy_from='../result/resnet32/flops_60_acts_80.pth'
    )
]

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
data = dict(samples_per_gpu=128, workers_per_gpu=2) # for single GPU