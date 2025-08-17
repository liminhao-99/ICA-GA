# ICA-driven Generative Attacks (ICA-GA)

## 准备工作

### 创建目录，获取源码

```sh
# 创建项目目录
mkdir ica_ga && cd ica_ga
# 拉取代码
git clone https://github.com/liminhao-99/ICA-GA.git
```

### 环境

```sh
conda create --name ica-ga python=3.12 -y
conda activate ica-ga
pip install -r requirements.txt
```

### 模型

[建议] 下载我们预训练好的模型文件，以快速复现实验。
- `ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth`: ResNet-50，基于PyTorch默认预训练参数，在Cifar100上充分微调。作为FL模型。
- `gen_1x_CT_RCSE_r50_fine_tuned.pth`: 论文中所使用的轻量级生成器，基于上述ResNet-50模型进行充分训练。
下载模型文件并放置于`./data/fl_models`及`./data/gen_models`:
```sh
# 创建数据目录
mkdir -p data/fl_models
mkdir -p data/gen_models
mkdir -p data/temp
# 下载预训练模型包
wget https://github.com/liminhao-99/ICA-GA/releases
# 解压
unzip -q data/temp/models_fine-tuned.zip -d data/temp
# 移动模型
mv data/temp/fl_models/ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth data/fl_models/ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth
mv data/temp/gen_models/gen_1x-CT-RCSE_r50-fine-tuned.pth data/gen_models/gen_1x-CT-RCSE_r50-fine-tuned.pth
```

### 数据集

- 攻击测试数据集`cifar100`等:，第一次运行实验时自动下载到`./data/datasets`。
- 生成器训练辅助数据集：需手动下载`ImageNet`。详见后续说明。

## 实验

命令行参数的格式是 `-key=value` 。必须用`=`分隔，且中间没有多余的空格。

确保当前工作目录是项目根目录，按以下指令启动实验。

以下只展示了典型的基本参数，可以复现我们论文中的实验。更多可调节参数，请阅读对应实验文件（如`fedsgd_attacks.py`）中的注释。

### FedSGD 攻击测试

```sh
python3 exps/fedsgd_attacks/fedsgd_attacks.py \
    -result_dir=results\
    -result_name=result.csv\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -fl_model_load.load_fc=False\
    -fl_model_kwargs.output_length=1000\
    -attacks.ICA-GA.model_save_dir=./data/gen_models\
    -attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned\
    -data_iter_kwargs.dataset_name=cifar100\
    -data_iter_kwargs.batch_size=32\
    -device=cuda:0\
    -seed=42\
    -data_iter_kwargs.dataloader_seed=42\
    -test_batchs=10\
    -rep_rate=0.0
```

| 基本参数 | 说明 |
|--|--|
|result_dir=results | 实验结果存放目录（相对于fedsgd_attacks.py） |
|result_name=result.csv | 实验结果输出文件名，csv文件 |
|fl_model_load.load_dir=./data/fl_models | FL模型加载目录（相对于工作目录，即实验根目录） |
|fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned | FL模型文件名 |
|fl_model_load.load_fc=False | FL模型不加载分类器（使用初始化参数） |
|fl_model_kwargs.output_length=1000 | 指定FL模型分类数 |
|attacks.ICA-GA.model_save_dir=./data/gen_models | 生成器模型加载目录 |
|attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned | 生成器模型文件名 |
|data_iter_kwargs.dataset_name=cifar100 | 测试数据集，可选 caltech256, cifar10, cifar100, mnist, imagenet |
|data_iter_kwargs.batch_size=32 | 测试批大小 |
|device=cuda:0 | 实验设备 |
|seed=42 | 主随机种子 |
|data_iter_kwargs.dataloader_seed=42 | 加载测试数据的随机种子 |
|test_batchs=10 | 实验重复次数 |
|rep_rate=0.0 | [可选]标签重复率：若设为0.0<rep_rate<=1.0，则构造特殊的数据批次，使得其中占比rep_rate的样本为相同标签 |

### FedAVG 攻击测试

```sh
python3 exps/fedavg_attacks/fedavg_attacks.py \
    -result_dir=results\
    -result_name=result.csv\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -fl_model_load.load_fc=False\
    -fl_model_kwargs.output_length=1000\
    -attacks.ICA-GA.model_save_dir=./data/gen_models\
    -attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned\
    -data_iter_kwargs.dataset_name=cifar100\
    -data_iter_kwargs.batch_size=8\
    -data_iter_kwargs.local_steps=4\
    -data_iter_kwargs.local_epochs=1\
    -test_batchs=10\
    -device=cuda:0\
    -seed=42\
    -data_iter_kwargs.dataloader_seed=42\
    -test_batchs=10\
    -device=cuda:0
```

| 基本参数 | 说明 |
|--|--|
|result_dir=results | 实验结果存放目录（相对于fedavg_attacks.py） |
|result_name=result.csv | 实验结果输出文件名，csv文件 |
|fl_model_load.load_dir=./data/fl_models | FL模型加载目录（相对于工作目录，即实验根目录） |
|fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned | FL模型文件名 |
|fl_model_load.load_fc=False | FL模型不加载分类器（使用初始化参数） |
|fl_model_kwargs.output_length=1000 | 指定FL模型分类数 |
|attacks.ICA-GA.model_save_dir=./data/gen_models | 生成器模型加载目录 |
|attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned | 生成器模型文件名 |
|data_iter_kwargs.dataset_name=cifar100 | 测试数据集，可选 caltech256, cifar10, cifar100, mnist, imagenet |
|data_iter_kwargs.batch_size=8 | 测试批大小 |
|data_iter_kwargs.local_steps=4 | 本地训练步数 |
|data_iter_kwargs.local_epochs=1 | 本地训练轮次 |
|test_batchs=10 | 实验重复次数 |
|device=cuda:0 | 实验设备 |
|seed=42 | 主随机种子 |
|data_iter_kwargs.dataloader_seed=42 | 加载测试数据的随机种子 |
|test_batchs=10 | 实验重复次数 |
|device=cuda:0 | 实验设备 |

### 生成器训练

生成器训练需要辅助数据集`imagenet`:。

请手动下载 [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) (`ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`) 并解压放置到`data/datasets`。正确放置的示例目录结构：

```
./data
└ datasets
    └ imagenet
        ├ ILSVRC2012_devkit_t12
        ├ train
        │  ├ n01440764
        │  └ ……
        └ val
            ├ n01440764
            └ ……
```

执行训练或增量微调：

```sh
python3 exps/train_generator/train_generator.py \
    -result_dir=results\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -generator_model_kwargs.save_dir=./data/gen_models\
    -generator_save_name=gen_1x-CT-RCSE_r50_1\
    -trainer_kwargs.train_dataset=imagenet\
    -trainer_kwargs.test_dataset=cifar100\
    -trainer_kwargs.batch_size=32\
    -trainer_kwargs.num_epochs=100\
    -trainer_kwargs.max_batchs=1000\
    -trainer_kwargs.lr=0.0001\
    -trainer_kwargs.loss.mse=1.0\
    -trainer_kwargs.loss.feat=0.01\
    -trainer_kwargs.loss.tv=0.01\
    -trainer_kwargs.loss.ssim=0.01\
    -trainer_kwargs.save_eval=psnr\
    -trainer_kwargs.save_eval_up=true\
    -trainer_kwargs.stop_eval_value=23\
    -trainer_kwargs.early_stop_patience=10
```

| 基本参数 | 说明 |
|--|--|
| result_dir=results | 训练记录保存目录（相对于train_generator.py）|
| recorder_save_name=result.csv | [可选]训练记录文件名。默认为 {generator_save_name}.csv|
| fl_model_load.load_dir=./data/fl_models | FL模型加载目录|
| fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned | FL模型文件名|
| generator_model_kwargs.save_dir=./data/gen_models | 生成器模型目录|
| generator_load_name=gen_1x-CT-RCSE_r50-fine-tuned | [可选]加载已有的生成器文件进行增量微调|
| generator_save_name=gen_1x-CT-RCSE_r50_1 | 生成器保存文件名|
| trainer_kwargs.train_dataset=imagenet | 生成器训练集|
| trainer_kwargs.test_dataset=cifar100 | 生成器评估集|
| trainer_kwargs.batch_size=32 | 训练批大小。24g显存时最大可设为48|
| trainer_kwargs.num_epochs=100 | 训练轮次上限|
| trainer_kwargs.max_batchs=1000 | [可选]单轮训练中使用的批次上限。降低该值可以减小训练粒度，更频繁的进行评估|
| trainer_kwargs.lr=0.0001 | 学习率。从头训练时`0.0001`，增量微调时`0.00001` |
| trainer_kwargs.loss.mse=1.0 | 损失权重|
| trainer_kwargs.loss.feat=0.01 ||
| trainer_kwargs.loss.tv=0.01 ||
| trainer_kwargs.loss.ssim=0.01 ||
| trainer_kwargs.save_eval=psnr | 主评估指标。当前轮次的评估值好于历史最佳值时，保存模型文件|
| trainer_kwargs.save_eval_up=true | 评估指标越大越好|
| trainer_kwargs.stop_eval_value=23 | [可选]早停评估值，指标达到该值时停止训练|
| trainer_kwargs.early_stop_patience=10 | [可选]早停耐心值，连续该轮次无提升时终止训练|

## 相关论文、项目

感谢

> Fast Generation-Based Gradient Leakage Attacks: An Approach to Generate Training Data Directly From the Gradient
- https://ieeexplore.ieee.org/abstract/document/10505158
- https://github.com/pigeon-dove/FGLA

> Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis
- https://arxiv.org/abs/2209.05578
- https://github.com/facebookresearch/cocktail_party_attack
