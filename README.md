# SR-SPECNet

Code and data for **“Model-Based Knowledge-Driven Learning Approach for Enhanced High-Resolution Automotive Radar Imaging.”**

**Dataset (Zenodo):** https://zenodo.org/records/16888438

---

## Table of Contents
1. [Dataset](#dataset)
2. [Environment](#environment)
3. [Data View](#data-view)
4. [Train](#train)
5. [Test](#test)
6. [Evaluate](#evaluate)
7. [Citation](#citation)

---

## Dataset

**Where:** Zenodo → https://zenodo.org/records/16888438  
**What to do:** Download the files from Zenodo and place them under your `data/BAMA/10/` directory.

**Expected layout (example)**
``` sh
project_root/
└── data/
    └── BAMA/
        └── 10/
            ├── train/
            └── test/
            └── Sample/
```
## Environment
```bash
conda env create -f environment.yml
conda activate SRSPECNet
```

## Data View
### Quick view (no saving)
```bash
python data_view.py --data_path ./data/BAMA --number_elements 10 --limit 3
```

### Save images (no GUI)
```bash
python data_view.py --data_path ./data/BAMA --number_elements 10 \
  --save_dir outputs/ra_pairs --no-show --limit 10
```

Expected outputs:
<p align="center">
  <img src="https://github.com/ruxinzh/SR-SPECNet/blob/main/outputs/ra_viz/ra_sample_00000.png" width="800">
</p>

## Train
```bash
python train.py \
  --data_path ./data/BAMA \
  --number_elements 10 \
  --output_size 256 \
  --learning_rate 1e-4 \
  --batch_size 4 \
  --epochs 50 \
  --k_folds 5 \
  --augmentation True \
  --loss True \
  --checkpoint_path ./checkpoint
```
## Test
```bash
python test.py \
  --data_path ./data/BAMA \
  --checkpoint_path ./checkpoint/<...>/fold_1_best_model_checkpoint.pth \
  --number_elements 10 \
  --output_size 256 \
  --batch_size 1 \
  --save_dir outputs/ra_pairs
```
## Evaluate
```bash
python eval.py \
  --data_path ./data/BAMA \
  --checkpoint_path ./checkpoint/<...>.pth \
  --number_elements 10 \
  --output_size 256 \
  --batch_size 1
```

Expected outputs:
<p align="center">
  <img src="https://github.com/ruxinzh/SR-SPECNet/blob/main/outputs/ra_pairs/ra_pair_00000.png" width="1000">
</p>

## Citation
If you use this work, please cite the following paper:
```BibTex
@article{zheng2025model,
  title   = {Model-Based Knowledge-Driven Learning Approach for Enhanced High-Resolution Automotive Radar Imaging},
  author  = {Zheng, Ruxin and Sun, Shunqiao and Liu, Hongshan and Chen, Honglei and Li, Jian},
  journal = {IEEE Transactions on Radar Systems},
  year    = {2025},
  publisher = {IEEE}
}
```



