# Multi-level Alignments for Compressed Video Super-Resolution

The overall training consists of three steps: training of the baseline model (train_baseline.py), shift prediction model (train_SPM.py), and unsupervised domain adaption (train_UDA.py). Each step requires updating the yml file according to the directory of the data, processing the specified panoramic video data, and training. The example yml files (train_baseline.yml, train_SPM.yml, and train_UDA.yml) corresponding to the three training steps are stored in the config folder.

## Environment:
```bash
conda create -n stdf python=3.7 -y && conda activate stdf

git clone --depth=1 https://github.com/ryanxingql/stdf-pytorch && cd stdf-pytorch/

# given CUDA 10.1
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## DCNv2:

```bash
cd ops/dcn/
bash build.sh
```
### Check if DCNv2 works (optional)

```bash
python simple_check.py
```

> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

## Dataset preparing:

### 1. Download the dataset through https://github.com/Archer-Tatsu/VQA-ODV.

### 2. Preparing data through the sequence extraction method:
```
python sequence_extraction.py
```
### 3. Generate LMDB data for training:
```
python create_lmdb.py --opt_path step1.yml
```

## Examples of instructions required during training (4, 2 and single GPUs):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train_step1.py --opt_path step1.yml
```
```
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train_step1 --opt_path step1.yml
```
```
#CUDA_VISIBLE_DEVICES=0 python train_step1 --opt_path step1.yml
```

## Test the trained model and obtain the corresponding indicator values:
```
python PanEnh.py 
```


