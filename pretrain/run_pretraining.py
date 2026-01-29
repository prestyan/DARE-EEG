# Training in 256Hz data and 4s
import os
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from engine_pretraining import *
from configs import *

torch.set_float32_matmul_precision("medium")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
seed_torch(2025)
# If you are not conducting ablation study, you can remove Module Selection or set it to Case5. 
model_save_name = f'DARE-EEG_3_{tag}_{Module_Selection}@'

# init model
# Use info in the config
print(f"Training MODEL-{tag} Module Selection: {Module_Selection}, CONFIGS:")
print(MODELS_CONFIGS[tag])
model = LitDAREEEG(get_config(**(MODELS_CONFIGS[tag])), 
                 USE_LOSS_A =(Module_Selection != "Case1"),
                 USE_LN     =(Module_Selection != "Case2"),
                 USE_SKIP   =(Module_Selection != "Case3"),
                 USE_LOSS_MC=(Module_Selection != "Case4"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
ckpt_cb = ModelCheckpoint(
    dirpath="./logs/checkpoints",                   
    filename=model_save_name+"{epoch}-{valid_loss:.4f}",   
    monitor="valid_loss", mode="min",       
    save_top_k=1,                           
    save_last=True,                     
    every_n_epochs=1                           
)
callbacks = [lr_monitor, ckpt_cb]
ckpt_path = "./logs/checkpoints/DARE-EEG_3_base@epoch=69-valid_loss=0.4626.ckpt"
trainer = pl.Trainer(accelerator="gpu", strategy='ddp_find_unused_parameters_true', devices=devices, max_epochs=max_epochs, callbacks=callbacks, 
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"ours_{tag}_{Module_Selection}_tb", version=2), 
                             pl_loggers.CSVLogger('./logs/', name=f"ours_{tag}_{Module_Selection}_csv")])
trainer.fit(model, train_loader, valid_loader)