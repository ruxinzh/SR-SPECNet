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
**What to do:** Download the files from Zenodo and place them under your `data` directory.

**Expected layout (example)**


**Download (example with wget)**
```bash
# Replace FILENAME with the exact file name from the Zenodo record page
wget -O data/BAMA/10/FILENAME "https://zenodo.org/records/16888438/files/FILENAME?download=1"

conda env create -f environment.yml
conda activate sr_doa

# Quick view (no saving)
python data_view.py --data_path ./data/BAMA --number_elements 10 --limit 3

# Save images (no GUI)
python data_view.py --data_path ./data/BAMA --number_elements 10 \
  --save_dir outputs/ra_pairs --no-show --limit 10

outputs/ra_pairs/ra_pair_00000.png

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

python test.py \
  --data_path ./data/BAMA \
  --checkpoint_path ./checkpoint/<...>/fold_1_best_model_checkpoint.pth \
  --number_elements 10 \
  --output_size 256 \
  --batch_size 1 \
  --save_dir outputs/ra_pairs

outputs/ra_pairs/ra_pair_00000.png

python eval.py \
  --data_path ./data/BAMA \
  --checkpoint_path ./checkpoint/<...>.pth \
  --number_elements 10 \
  --output_size 256 \
  --batch_size 1

```BibTex
@article{zheng2025model,
  title   = {Model-Based Knowledge-Driven Learning Approach for Enhanced High-Resolution Automotive Radar Imaging},
  author  = {Zheng, Ruxin and Sun, Shunqiao and Liu, Hongshan and Chen, Honglei and Li, Jian},
  journal = {IEEE Transactions on Radar Systems},
  year    = {2025},
  publisher = {IEEE}
}
```



