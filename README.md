Official repository of the GLiRA: Black-Box Membership Inference Attack via Knowledge Distillation.


## Quick Start

1. Create a virtual environment and activate it (e.g conda environment)
```
conda create -n glira python=3.10
conda activate glira
```
2. Install Pytorch and torchvision following the official instructions
```
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
3. Install build requirements:
```
pip install -r requirements.txt
```

## Training

Here, we provide an example to obtain results for CIFAR10 dataset for target-shadow pair (ResNet34, ResNet34).

1. Train the target model:
```
python train.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --noamp
```
After training, you can find the model weights at `./checkpoints/cifar10/target/res34.pth`.

2. Train shadow models with standard Cross-Entropy Loss:
```
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128 --noamp
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34/`.

3. Train shadow models by distillation using Kullback-Leibler Divergence Loss (GLiRA (KL)):
```
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --target_net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128 --lambd 0 --noamp
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34_1.0dis_res34/`.

4. Train shadow models by distillation using Mean-Squared-Error Loss (GLiRA (MSE)):
```
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --target_net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128 --mse_distillation --noamp
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34_MSEdis_res34/`.

## Evaluation

Here, we provide an example how to evaluate the success rate of the attacks obtained in the previous paragraph.

All the results are stored at `./results/cifar10/results.pickle` file, which is a `pd.DataFrame` instance. Column `method` contains attack method.

1. Obtain the *Offline LiRA* results:
```
python inference.py --shadow_net res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
```
2. Obtain the *GLiRA (KL)* results:
```
python inference.py --shadow_net res34_1.0dis_res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
```
3. Obtain the *GLiRA (MSE)* results:
```
python inference.py --shadow_net res34_MSEdis_res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
