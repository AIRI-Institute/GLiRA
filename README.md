Official repository of the GLiRA: Black-Box Membership Inference Attack via Knowledge Distillation.


## Installation

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
```bash
python train.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100
```
After training, you can find the model weights at `./checkpoints/cifar10/target/res34.pth`.

2. Train shadow models with standard Cross-Entropy Loss (LiRA):
```bash
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34/`.

3. Train shadow models by distillation using Kullback-Leibler Divergence Loss (GLiRA (KL)):
```bash
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --target_net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128 --lambd 0
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34_1.0dis_res34/`.

4. Train shadow models by distillation using Mean-Squared-Error Loss (GLiRA (MSE)):
```bash
python train_experiment.py --lr 0.1 --weight_decay 5e-4 --opt sgd --net res34 --target_net res34 --dataset cifar10 --per_model_dataset_size 20000 --bs 256 --size 32 --n_epochs 100 --num_shadow 128 --mse_distillation --warmup_epochs 5
```
Checkpoints will be stored at `./checkpoints/cifar10/shadow/res34_MSEdis_res34/`.

**NOTE:** Training the model from scratch using MSE loss can lead to divergence in some scenarios. To mitigate this issue, you can either train with a smaller learning rate (`--lr 0.01`), or train with a standard Cross-Entropy Loss on first epochs (`--warmup_epochs 5`) and than switch to the MSE distillation.

## Evaluation

Here, we provide an example how to evaluate the success rate of the attacks obtained in the previous paragraph.

All the results are stored at `./results/cifar10/results.pickle` file, which is a `pd.DataFrame` instance. Column `method` contains attack method.

**NOTE:** By default we first cache the evaluation data (`--cache_data`) to exclude randomness introduced by augmentations between different runs. For the considered datasets, the required memory is small, but if you still don't want to store them just disable this flag.

If `--cache_data` is specified, cached evaluation data is stored at `./eval_data`.

1. Obtain the *Offline LiRA* results:
```bash
python inference.py --shadow_net res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
```
2. Obtain the *GLiRA (KL)* results:
```bash
python inference.py --shadow_net res34_1.0dis_res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
```
3. Obtain the *GLiRA (MSE)* results:
```bash
python inference.py --shadow_net res34_MSEdis_res34 --target_net res34 --num_shadow 128 --dataset cifar10 --num_aug 10 --n_samples 20000 --evaluation_objective stable_logit --evaluation_type lira --fix_variance --cache_data
```

To view the metrics, you can do the following:

```python
import pandas
data = pandas.read_pickle('./results/results.pickle')
data[['method', 'shadow_net', 'target_net', 'dataset', 'auc', 'acc', 'tpr@fpr']]
```

The True-Positive-Rates are reported for the next fixed False-Positive-Rates: $0.01$%, $0.1$%, $1$%, $10$%
