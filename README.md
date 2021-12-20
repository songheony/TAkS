# [No Regret Sample Selection with Noisy Labels (TAkS)](https://arxiv.org/abs/2003.03179)

![Figure](assets/Fig1.png?raw=true "Figure")

> Deep neural networks (DNNs) suffer from noisy-labeled data because of the risk of overfitting. To avoid the risk, in this paper, we propose a novel DNN training method with sample selection based on adaptive k-set selection, which selects k (< n) clean sample candidates from the whole n noisy training samples at each epoch. It has a strong advantage of guaranteeing the performance of the selection theoretically. Roughly speaking, a regret, which is defined by the difference between the actual selection and the best selection, of the proposed method is theoretically bounded, even though the best selection is unknown until the end of all epochs. The experimental results on multiple noisy-labeled datasets demonstrate that our sample selection strategy works effectively in the DNN training; in fact, the proposed method achieved the best or the second-best performance among state-of-the-art methods, while requiring a significantly lower computational cost.

## Implementations

We've implemented baselines and the proposed method in this repository.

* [F-correction](https://arxiv.org/abs/1609.03683)[<https://github.com/giorgiop/loss-correction>]
* [Decouple](https://arxiv.org/abs/1706.02613)[<https://github.com/emalach/UpdateByDisagreement>]
* [Co-teaching](https://arxiv.org/abs/1804.06872)[<https://github.com/bhanML/Co-teaching>]
* [Co-teaching+](https://arxiv.org/abs/1901.04215)[<https://github.com/bhanML/coteaching_plus>]
* [JoCoR](https://arxiv.org/abs/2003.02752)[<https://github.com/hongxin001/JoCoR>]
* [Class2Simi](https://arxiv.org/abs/2006.07831)[<https://github.com/scifancier/Class2Simi>]
* [TV](https://arxiv.org/abs/2102.02414)[<https://github.com/YivanZhang/lio/tree/master/ex/transition-matrix>]
* [CDR](https://openreview.net/forum?id=Eql5b1_hTE4)[<https://github.com/xiaoboxia/CDR>]
* [TAkS](https://arxiv.org/abs/2003.03179) (Ours)

## Requirements

We strongly recommend using Anaconda or Docker.

### Anaconda

#### For training and test

Our training and testing code depends solely on PyTorch (>= 0.4.1).  
Additionally, a tensorboard is used to visualize the loss during learning.  

```sh
conda create -n TAkS python=[PYTHON_VERSION]
conda activate TAkS
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
conda install tensorboard
```

Note that all commands should be executed in the corresponding console  
or after entering to the conda environment through the following command in a new console.

```sh
conda activate TAkS
```

#### For visualization

Note that cuml is only used for visualization_minor.py and it requires Python version 3.7.

```sh
conda install pandas
conda install -c plotly plotly plotly-orca
conda install -c rapidsai cuml
```

### Docker

We provide pre-built docker image and you can easily pull and use that image.

```sh
docker pull songheony/taks
```

If you want to build custom image, you can make it based on `docker/Dockerfile`.

## How to run

The simplest way to train a model is as follows.

```sh
python main.py --method_name [METHOD_NAME] --dataset_name [DATASET_NAME] --noise_type [NOISE_TYPE] --noise_ratio [NOISE_RATIO]
```

METHOD_NAME must be one of the following values:  
f-correction, decouple, co-teaching. co-teaching+, jocor, class2simi, tv, cdr, and ours.

DATASET_NAME must be one of the following values:  
mnist, cifar10, cifar100, and clothing1m.

NOISE_TYPE must be one of the following values:  
symmetric and asymmetric.

For example, how to train TAkS on CIFAR-10 with Symmetric-80% is as follows.

```sh
python ./main.py --method_name "ours" --k_ratio 0.2 --lr_ratio 0.005 --dataset_name "cifar10" --noise_type "symmetric" --noise_ratio 0.8
```

If you want to train F-correction, it is necessary to train Standard with validation set first.  

```sh
python ./main.py --method_name "standard" --dataset_name "cifar10" --noise_type "symmetric" --noise_ratio 0.8 --train_ratio 0.8
python ./main.py --method_name "f-correction" --dataset_name "cifar10" --noise_type "symmetric" --noise_ratio 0.8
```

### Visualize progress of training with tensorboard

You can see the loss and accuracy graph of a model during training.

```sh
tensorboard --logdir logs/
```

## Reproduce our results

You can reproduce our results by following commands.  

```sh
# Run baselines and TAkS on MNIST, CIFAR-10, and CIFAR-100
bash scripts/run_datasets.sh [SEED_NUMBER] [GPU_NUMBER]

# Run baselines and TAkS on Clothing1M
bash scripts/run_clothing.sh [SEED_NUMBER] [GPU_NUMBER]

# Run ablation study
bash scripts/run_ablation.sh [SEED_NUMBER] [GPU_NUMBER]

# Run baselines with fixed precisions
bash scripts/run_precision.sh [SEED_NUMBER] [GPU_NUMBER]

# Visualize results
python visualize.py

# Visualize Figure 3 and Figure 4 in our paper
python visualize_minor.py
```

|                 | Standard | F-correction | Decouple |               |          Co-teaching          |           |          Co-teaching+         |           |              JoCoR             |           |          TAkS(Ours)          |               |
|:---------------:|:--------:|:------------:|:--------:|:-------------:|:-----------------------------:|:---------:|:-----------------------------:|:---------:|:------------------------------:|:---------:|:-----------------------------:|:-------------:|
|                 |    Acc   |      Acc     |    Acc   |   xT   |              Acc              | xT |              Acc              | xT |               Acc              | xT |              Acc              |   xT   |
| **MNIST**           |          |              |          |               |                               |           |                               |           |                                |           |                               |               |
|  Symmetric-20% |   78.67  |     90.71    |   94.74  | **1.10** |             94.52             |    1.60   |             97.77             |    1.61   |  **97.88** |    1.47   |  **97.94** |      1.19     |
|  Symmetric-50% |   51.22  |     77.36    |   66.77  |      1.46     |             89.50             |    1.59   |             95.67             |    1.67   |  **95.90** |    1.49   |  **97.17** | **0.97** |
|  Symmetric-80% |   22.43  |     51.16    |   27.42  |      1.50     |             78.52             |    1.59   |             66.13             |    1.73   |  **88.53** |    1.49   |  **92.32** | **0.81** |
| Asymmetric-40% |   78.97  |     88.99    |   82.04  |      1.35     |             90.21             |    1.59   |             92.48             |    1.65   |  **93.91** |    1.50   |  **95.77** | **1.22** |
| **CIFAR-10**        |          |              |          |               |                               |           |                               |           |                                |           |                               |               |
|  Symmetric-20% |   68.92  |     74.21    |   69.95  |      1.61     |             78.16             |    1.98   |             78.68             |    2.00   |  **85.75**  |    1.73   | **83.90** | **0.99** |
|  Symmetric-50% |   41.93  |     52.68    |   40.91  |      1.71     |             70.79             |    1.97   |             56.90             |    1.99   |  **78.92**  |    1.73   | **76.83** | **0.74** |
|  Symmetric-80% |   15.85  |     18.99    |   15.29  |      1.82     | **26.54** |    1.98   |             23.50             |    2.00   |              25.51             |    1.73   |  **40.24** | **0.53** |
| Asymmetric-40% |   69.23  |     69.64    |   69.10  |      1.51     | **73.59** |    1.99   |             68.45             |    2.00   |  **76.13**  |    1.74   |             73.43             | **1.04** |
| **CIFAR-100**       |          |              |          |               |                               |           |                               |           |                                |           |                               |               |
|  Symmetric-20% |   35.51  |     36.04    |   33.82  |      1.75     |             44.03             |    1.97   |             49.24             |    1.99   |  **53.10**  |    1.73   | **50.74** | **0.93** |
|  Symmetric-50% |   17.31  |     21.14    |   15.81  |      1.82     |             34.96             |    1.97   |             40.26             |    2.00   |  **43.28**  |    1.73   | **40.98** | **0.68** |
|  Symmetric-80% |   4.25   |     7.48     |   4.03   |      1.90     | **14.81** |    1.97   |             13.99             |    2.00   |              12.90             |    1.72   |  **16.03** | **0.52** |
| Asymmetric-40% |   27.91  |     27.11    |   26.95  |      1.75     |             28.69             |    1.98   | **34.30** |    2.01   |              32.39             |    1.74   |  **35.23** | **0.98** |
| **Clothing1M**      |          |              |          |               |                               |           |                               |           |                                |           |                               |               |
|       Best      |   67.62  |     67.34    |   68.32  |      1.97     |             68.37             |    1.99   |             68.51             |    1.99   |  **70.30** |    2.00   | **70.28** |      **0.78**     |
|       Last      |   66.05  |     66.73    |   67.69  |               |             68.12             |           |             68.51             |           | **69.79** |           |  **70.28** |               |

## Run TAkS on your custom dataset

This repository follows the standard PyTorch code.  
Therefore, it is quite easy to apply TAkS to your custom dataset.

First, add your custom dataset to [dataset.py](./dataset.py).  
Second, copy [tune_clothing.sh](scripts/tune_clothing.sh) and [run_clothing.sh](scripts/run_clothing.sh) to create hyperparameter tuning and training scripts for your custom dataset.  
Third, run following commands:

```sh
# Tune hyperparamers
base scripts/[TUNING_CODE] [SEED_NUMBER] [GPU_NUMBER]

# Run TAkS based on tuned hyperparameters
base scripts/[TRAINING_CODE] [SEED_NUMBER] [GPU_NUMBER]
```
