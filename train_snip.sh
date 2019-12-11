CUDA_VISIBLE_DEVICES=0 python3 train.py --experiment 'cifar10_snip_vgg16_bn_1' --lr 1e-1 --max-epoch 200 --lr-decaysteps 100 150 180 --model_name 'vgg16_bn' --kappa 0.70
