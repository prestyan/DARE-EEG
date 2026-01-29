# DARE-EEG: A Foundation Model for Mining Dual-Aligned Representations of EEG

This repository provides the official implementation of **DARE-EEG**, a foundation model designed to learn **dual-aligned EEG REpresentations** through large-scale pre-training and adaptation to diverse downstream tasks.

We have for the first time proposed the mask-invariance property of EEG in the encoding space. Based on this, DARE-EEG focuses on robust representation learning under missing, masked, and noisy EEG signal conditions, which are common in non-invasive EEG acquisition. The learned representations are transferable across multiple EEG paradigms and datasets.

---

## Repository Structure

The repository is organized as follows:

```text
DARE-EEG/
├── pretrain/
│   └── logs/
│       ├── checkpoints/
│       └── tensorboard/
├── downtasks/
├── downtasks_others/
└── data/
│   └── pretrain/
│   └── downtasks/
```

## DARE-EEG/pretrain
This directory contains the pre-training framework and running code for DARE-EEG, including:

- Model architecture definitions

- Mask Alignment and Anchor Alignment modules

- Pre-training configuration files and training scripts

All training logs, model checkpoints, and other visualization files are saved under:
```bash
pretrain/logs
pretrain/logs/checkpoints
```

## DARE-EEG/downtasks
This directory contains code for running downstream tasks on the TUAB and TUEV datasets.

## DARE-EEG/downtasks_others
This directory provides example implementations for additional downstream EEG tasks. Currently, an example on the BCIC-2B dataset is included to demonstrate the transferability of DARE-EEG to motor imagery classification tasks.

This directory can also serve as a template for extending DARE-EEG to new EEG datasets and paradigms.

## DARE-EEG/data
This directory contains dataset-related scripts and documentation, including:

- Data preprocessing and format conversion scripts

- Dataset organization examples

- Dataset usage instructions and citation information

Your processed pre-training data should be placed in:

```bash
cd data/pretrain/merged/TrainFolder/
cd data/pretrain/merged/ValidFolder/
```

> **Note**  
> Raw EEG data are **NOT included** in this repository due to copyright and privacy restrictions.  
> Please download the datasets from their official sources.

## Getting Started
### Environment Setup
Please install the requirements.txt using `pip install -r requirements.txt`:
```text
apex==0.9.10dev
braindecode==1.3.2
deepspeed==0.18.2
einops==0.8.2
h5py==3.10.0
linear_attention_transformer==0.19.1
matplotlib==3.8.4
mne==1.11.0
moabb==1.4.3
numpy==2.4.1
pandas==1.5.3
parse==1.20.2
pyhealth==1.1.6
pytorch_lightning==2.5.2
PyYAML==6.0.1
PyYAML==6.0.3
scikit_learn==1.3.0
scipy==1.17.0
tensorboardX==1.8
tensorboardX==2.6.4
thop==0.1.1.post2209072238
timm==0.6.13
torch==2.2.2
torchaudio==2.2.2
torcheeg==1.1.3
torchvision==0.17.2
tqdm==4.66.2
```
### Pre-training

To start pre-training DARE-EEG, first, you need to prepare the training dataset.  Instructions for downloading and setting up the dataset can be found in Appendix D of the paper. After downloading the raw data, please run `python ./data/pretrain/prepare_pretrain_dataset.py`. After the process is complete, the file structure should be as follows:
```bash
data/pretrain
└── merged
    ├── TrainFolder
    │   ├── 0
    │   ├── 1
    │   ├── 2
    │   ├── 3
    │   └── 4
    └── ValidFolder
        ├── 0
        ├── 1
        ├── 2
        ├── 3
        └── 4
```
Then, run the code `python run_pretraining` in the pretrain folder. If you want to change the model architecture, please modify the `tag` option in `config.py`.
