import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

from Modules.models.base_model import EEGTransformer
from Modules.models.prob_model import ConvHead

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics

use_channels_names = [      
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]

class LitModelConvp(pl.LightningModule):

    def __init__(self, fold, tag="deep",load_path="../pretrain/logs/checkpoints/DARE-EEG_3_large@epoch=193-valid_loss=0.6080.ckpt"):
        super().__init__()    
        self.chans_num = 19
        # load_path = ''
        self.tag = tag
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
            patch_size=32*2,
            embed_num=embed_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.fold = fold
        use_channels_names = [ 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6' ]
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        # self.chan_conv       = Conv1dWithConstraint(22, self.chans_num, 1, max_norm=1)
        
        self.chan_conv = ConvHead(
            in_channels=3,
            out_channels=7,
            in_time_length=1024,
            out_time_length=1024,
            conv_layers=1,
            dropout=0.
        )
        
        self.linear_probe1   =   LinearWithConstraint(feature_dimension, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(16*16, 2, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.best_epoch = -1
        self.best_metrics = {}
        self.is_sanity=True
        
    def forward(self, x):
        x = x/10
        
        x = self.chan_conv(x)
        
        self.target_encoder.eval()
        
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        
        h = h.flatten(1)
        
        h = self.linear_probe2(h)
        
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # y = F.one_hot(y.long(), num_classes=4).float()
        
        label = y
        label = y.long().view(-1)
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
    
    
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)

        if self.best_epoch == -1 or results['balanced_accuracy'] > self.best_metrics.get('balanced_accuracy', 0):
            self.best_epoch = self.current_epoch
            self.best_metrics = results
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)

        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        return loss
    
    def on_fit_end(self):
        print(f"Best epoch: {self.best_epoch}")
        for metric_name, metric_value in self.best_metrics.items():
            print(f"Best val_{metric_name}: {metric_value:.4f}")
        
        with open(f'./logs/para_metrics_bcic2b_{self.tag}.txt', 'a') as f:
            kv_val = " ".join([f"{k}={v:.4f}" for k, v in self.best_metrics.items()])
            f.write(f"Fold {self.fold}\tBestEpoch={self.best_epoch}\tValid: {kv_val}\n")
    
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 

# load configs
from utils import get_data
data_path = "../data/downtasks/BCIC_2b_0_38HZ"
import math
# used seed: 7
seed_torch(2024)
for i in range(1, 10):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1,is_few_EA = True, target_sample=1024)
    
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=0, shuffle=False)
    
    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 5e-4

    # init model
    model = LitModelConvp(fold=i, tag="small", load_path="../pretrain/logs/checkpoints/DARE-EEG_3_small@epoch=168-valid_loss=0.6119.ckpt")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         devices=[3,], 
                         precision=32,
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         num_sanity_val_steps=0,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"ours_BCIC2B_{model.tag}_tb", version=f"subject{i}_{i}"), 
                                 pl_loggers.CSVLogger('./logs/', name=f"ours_BCIC2B_{model.tag}_csv", version=f"subject{i}")])

    trainer.fit(model, train_loader, test_loader, ckpt_path='last')