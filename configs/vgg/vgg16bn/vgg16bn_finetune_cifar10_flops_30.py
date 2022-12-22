_base_ = [
    '../../_base_/models/vgg16bn_cifar10.py', '../../_base_/datasets/cifar10_bs128.py',
    '../../_base_/schedules/cifar10_bs128_2.py', '../../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='FisherPruningHook',
         pruning=False,
         deploy_from='../result/vgg16/flops_30_acts_57.pth')
]

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
data = dict(samples_per_gpu=128, workers_per_gpu=2) # for single GPU
