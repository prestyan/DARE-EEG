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
downtasks/logs
downtasks_others/logs
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

## Get Checkpoints

You can access the pre-trained weights in the [`checkpoints`](pretrain/logs/checkpoints/README.md) folder; more weights will be updated there.

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
Then, run the code `python run_pretraining` in the pretrain folder. If you want to change the model architecture, please modify the `tag` option in `config.py`. The code also includes a Case option, which is used for module ablation experiments and should generally be set to Case5.

### Downtasks: TUAB & TUEV
First, you need to download the TUH-EEG dataset, selecting both the TUAB and TUEV categories.  Refer to the download links provided in Appendix D of the paper. After downloading, organize the data as follows, with the `processed` folder used to store the processed data.

- RUN TUAB
```bash
data/downtasks/TUAB
└── tuh_eeg_anbormal
    ├── AAREADME.txt
    ├── AAREADME.txt,v
    ├── edf
    │   ├── eval
    │   ├── processed
    │   └── train
    ├── needs_fixin.list
    ├── xx.dat
    └── xx.sh
```
Then run `python process_tuab.py`. The processed data will be divided into training, validation, and test sets and placed in the `processed` folder. Finally, please run the following code to load the pre-trained encoder and test it on the TUAB dataset. The complete command we recommend running is as follows:
```bash
cd DARE-EEG/downtasks/
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
probe_tueg.py --dataset=TUAB --data_path_1=../data/downtasks/TUAB/tuh_eeg_abnormal/edf/processed --encoder_path=../pretrain/logs/checkpoints/DARE-EEG_3_Deep@epoch=193-valid_loss=0.6080.ckpt
```
- RUN TUEV
You need to organize the TUEV data in the following format. Then run `python process_tuev.py`.
```bash
data/downtasks/TUEV
└── tuh_eeg_events
    ├── AAREADME.txt
    ├── AAREADME.txt,v
    ├── edf
    │   ├── eval
    │   ├── processed
    │   └── train
    ├── needs_fixin.list
    ├── xx.dat
    └── xx.sh
```
The processed data will be divided into training, validation, and test sets and placed in the `processed` folder. Finally, please run the following code to load the pre-trained encoder and test it on the TUAB dataset. The complete command we recommend running is as follows:
```bash
cd DARE-EEG/downtasks/
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500
probe_tueg.py --enable_deepspeed --layer_decay=0.65 --dropout=0.3 --max_epochs=30 --dataset=TUEV --data_path_1=../data/downtasks/TUEV/tuh_eeg_events/edf/processed --encoder_path=../pretrain/logs/checkpoints/DARE-EEG_3_Deep@epoch=193-valid_loss=0.6080.ckpt
```
If you encounter errors with the PyTorch kernel on multiple GPUs during runtime, we recommend training on a single GPU. This is due to incompatibility between the operators and the CUDA graphics card configuration.  Reducing the batch size may also help.

> **Note**  
> Please always pay attention to the file storage location; if adjustments are needed, be sure to modify the file path to avoid errors.
> On the TUEV dataset, we used different label weights to prevent the majority class samples from dominating the loss function, but this operation is not necessary for the TUAB dataset.

### Downtasks: An example on BCIC-2B

We first provide example code for an experiment using DARE-EEG pre-trained weights and a conv-linear-probing on the BCIC-2B dataset. This code can be run directly. 
First, please download the BCIC-2B dataset. Organize it into the following format:
```bash
DARE-EEG/data/downtasks/BCIC
├─raw_data
|  +---BCI-2b
|  |       B0101T.gdf
|  |       B0101T.mat
|  |       B0102T.gdf
|  |       B0102T.mat
|  |       ...
├─Data
```
The `Data` folder is used to store the processed data. Then, run the following command to obtain the preprocessed data:
```bash
cd DARE-EEG/data/downtasks
python process_bcic.py
```
The `process_bcic.py` file also contains code for processing the BCIC-2A dataset; simply uncomment the relevant lines in the main function to use it.
Then, run the following commands to use and freeze the pre-trained weights and then conduct experiments on the BCIC-2B dataset.
```bash
cd DARE-EEG/downtasks_others
python convp_bcic2b.py
```
Additionally, the `convp_bcic2b_para.py` file is provided, which is code for parameter ablation experiments. If you need to use different pre-trained weights, please change the `tag` option in `LitModelConvp`.

## How to use it on other expanded datasets

We provide a paradigm for applying DARE-EEG to other EEG datasets. The specific steps are as follows.

1. **Copy the required libraries**: To use DARE-EEG fully, the following two files are required: `base_model.py` and `probe_model.py`. They are stored in `downtasks_others/Modules/models`. `base_model` is used to define the EEG encoder, while `probe_model` is used to define the channel and sampling rate adaptation module, as well as the dataset-specific task head of the model.
2. **Define the required models in the code**: You can define your dataset-specific model using the following code:
```python
from models.base_model import EEGTransformer
from models.prob_model import ConvHead
...
def load_DARE_EEG(load_path = '../pretrain/logs/checkpoints/DARE-EEG_3_large@epoch=193-valid_loss=0.6080.ckpt',
                  tag = 'deep
                  load_pretrain = True,
                  img_size,
                  use_channels_names, # The channel name you are using...
                  in_channels, # Dataset input channel
                  out_channels, # =len(use_channels_names)
                  in_time_length,
                  out_time_length,
                  conv_layers):

        # init model
        if tag=="deep":
            embed_num, embed_dim, depth, num_heads = 4, 512, 8, 4
            feature_dimension = 2048
        elif tag=="base":
            embed_num, embed_dim, depth, num_heads = 4, 256, 8, 8
            feature_dimension = 1024
        elif tag=="small":
            embed_num, embed_dim, depth, num_heads = 1, 256, 6, 4
            feature_dimension = 256
        elif tag=="light":
            embed_num, embed_dim, depth, num_heads = 1, 128, 6, 4
            feature_dimension = 128
        elif tag=="nano":
            embed_num, embed_dim, depth, num_heads = 1, 64, 2, 4
            feature_dimension = 64

        target_encoder = EEGTransformer(
            img_size=[7, 1024],
            patch_size=64,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        if load_pretrain:
            pretrain_ckpt = torch.load(load_path)
            
            target_encoder_stat = {}
            for k,v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]]=v
                
            self.target_encoder.load_state_dict(target_encoder_stat)
            for blk in model.blocks:
                for p in blk.parameters():
                    p.requires_grad = False  
        chan_conv = ConvHead(
            in_channels=in_channels,
            out_channels=out_channels,
            in_time_length=in_time_length,
            out_time_length=out_time_length,
            conv_layers=conv_layers,
            dropout=0.1
        )
    return target_encoder, chan_conv, chans_id
```
3. **Using a predefined model**: Below, you can use it in the `forward` method like this:
```python
encoder, chead, chans_id = load_DARE_EEG(...)
...
# in model forward:
def forward(self, x, chans_id):
    x = chead(x)
    x = encoder(x, chans_id.to(x))
    # The subsequent structure you use to receive the encoder's output representations.
    ...    
```















