# HiDFD

Hybrid Data-Free Knowledge Distillation
# Toolbox for HiDFD

This repository aims to provide a compact and easy-to-use implementation of our proposed HiDFD on a series of data-free knowledg distillation tasks. 

- Computing Infrastructure:
  - We use one NVIDIA V100 GPU for CIFAR experiments and use one NVIDIA A100 GPU for ImageNet experiments. The PyTorch version is 1.12.

## Get the pretrained teacher models

```bash
# CIFAR10
python train_teacher.py --dataset cifar10

# ImageNet
python train_teacher.py --batch_size 256 --dataset imagenet --model ResNet18 --num_workers 32 --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23333 --multiprocessing-distributed --dali gpu --trial 0
```
## Generative distillation
Cd to generative_distillation
```bash
run sh train.sh
```
## Student distillation
Cd to student_distillation
```bash
# CIFAR10
python train_student.py --path_t './save/teachers/models/**.pth' --repeat_num 10
# ImageNet
python train_student.py --path_t './save/teachers/models/**.pth' \
 --path_train /hss/giil/temp/data/web_resized/imgnet \
 --batch_size 256 --num_workers 16 --gpu_id 0,1,2,3,4,5,6,7 \
 --dist-url tcp://127.0.0.1:23444 --multiprocessing-distributed
```





