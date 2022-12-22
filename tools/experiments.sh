# IFSO pruning ResNet-32/VGG-16 on CIFAR-10

#step1: pre-train
#python train_fullnetwork_resnet32.py --config='../configs/resnet/resnet32_8xb16_cifar10.py' --work-dir='../result/temp'

#python train_fullnetwork_vgg16bn.py --config='../configs/vgg/vgg16bn_8xb16_cifar10.py' --work-dir='../result/temp'

#step2: pruning
#python train_pruning_conv_fc_norm_sqrt.py --config='../configs/resnet/resnet32/resnet32_pruning_cifar10.py' --work-dir='../result/temp'

#python train_pruning_conv_fc_vgg_norm_sqrt.py --config='../configs/vgg/vgg16bn/vgg16bn_pruning_cifar10.py' --work-dir='../result/temp'

#step3: finetune

#python train_finetune_conv_fc_norm_sqrt.py --config='../configs/resnet/resnet32/resnet32_finetune_cifar10_flops_50.py'  --work-dir='../result/temp'

#python train_finetune_conv_fc_vgg_norm_sqrt.py --config='../configs/vgg/vgg16bn/vgg16bn_finetune_cifar10_flops_50.py'  --work-dir='../result/temp'



