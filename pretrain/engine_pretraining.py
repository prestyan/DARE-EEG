import os
import math
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import copy
import torchvision
from pytorch_lightning import loggers as pl_loggers


from utils import WarmupCosineSchedule, CosineWDSchedule, grad_logger
from modeling_pretraining import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from mcontra_learning import MaskContrastiveLearning
from configs import *

# Channels used in pretraining 10-20 standard system
use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]


class LitDAREEEG(pl.LightningModule):

    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True, USE_LOSS_MC=True):
        super().__init__()    
        self.USE_LOSS_A  = USE_LOSS_A
        self.USE_LN      = USE_LN
        self.USE_SKIP    = USE_SKIP
        self.USE_LOSS_MC = USE_LOSS_MC
        
        # EEG Encoder
        encoder = EEGTransformer(
            img_size=[58, 1024],
            patch_size=64,
            init_std=0.02,
            interpolate_factor=2.,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])
        
        # EEG Predictor
        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])
        
        # EEG Reconstructor
        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])
        
        # For Anchor Alignment
        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
            
        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)
        
        self.loss_fn        = torch.nn.MSELoss()

        # Mask Contrastive Learning
        # For Mask Alignment
        self.loss_mc        = MaskContrastiveLearning(models_configs) 
        
    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        
        C, N = num_patchs
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y: # There is a probability of 1 - p_n_y that only a portion of the channels of patch-i at that time point will be selected.
                    mask_x.append(c_idx[:mC_x]) # Select mC_x channels as input (context).
                    mask_y_bx.append(c_idx[mC_x:]) # The remaining channels can be used as supplementary candidates for mask_y.
                else: # JUMP patch-i
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0) 
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0)

    def derive_view_with_target_iou(self, mask_x1, C, N, target_iou=(0.2, 0.8), jitter=0.05):
        """
        `mask_x1`: [N, m] LongTensor, flattened indices of visible channels for each time patch in view 1.
        Returns `mask_x2`, also of size [N, m], such that the IoU falls within the target range.
        """
        Bm, m = mask_x1.shape  
        U = C * N              
        low, high = target_iou

        import random
        rho = random.uniform(low, high)
        s = (2 * rho) / (1 + rho)      # Sharing ratio
        s = max(0.0, min(0.95, s + random.uniform(-jitter, jitter)))  # Slight shaking
        k_shared = max(0, min(m, int(round(s * m))))
        k_new = m - k_shared

        mask_x2 = []
        for i in range(Bm):
            # The complete index range of this patch.
            base = i * C
            all_i = torch.arange(base, base + C, device=mask_x1.device)
            v1 = mask_x1[i]                               

            if k_shared > 0:
                share_idx = v1[torch.randperm(m, device=v1.device)[:k_shared]]
            else:
                share_idx = torch.empty(0, dtype=v1.dtype, device=v1.device)

            if k_new > 0:
                cand = all_i[~torch.isin(all_i, v1)]
                pick = cand[torch.randperm(cand.shape[0], device=cand.device)[:k_new]]
                v2 = torch.cat([share_idx, pick], dim=0)
            else:
                v2 = share_idx
            mask_x2.append(v2)
        mask_x2 = torch.stack(mask_x2, dim=0)  # [N, m]
        return mask_x2
    
    def create_contrastive_views(self, x, mask_x1=None, target_iou=(0.3, 0.6), max_tries=10):
        """Create two different masked views for contrastive learning"""
        # Create two different sets of masks for the same sample.
        # C N (Chaneel, N_patches)
        num_patches = self.encoder.num_patches
        C, N = num_patches
        device = x.device
        best = None
        best_gap = float('inf')

        if mask_x1 != None:
            mask_x2 = self.derive_view_with_target_iou(mask_x1, C, N)
            return mask_x1, mask_x2
        
        for attempt in range(max_tries):
            if mask_x1 == None:
                print("[WARNING: detect mask_x1 is none] Generate new mask_x1...")
                mask_x1, mask_y1 = self.make_masks(num_patches)
            mask_x2 = self.derive_view_with_target_iou(mask_x1, C, N)

            total = C * N
            visible1 = torch.zeros(total, dtype=torch.bool, device=device)
            visible2 = torch.zeros(total, dtype=torch.bool, device=device)

            # mask_x*: [N, mC_x]
            visible1[mask_x1.reshape(-1)] = True
            visible2[mask_x2.reshape(-1)] = True

            inter = (visible1 & visible2).sum().item()
            union = (visible1 | visible2).sum().item()
            if union == 0:
                continue
            iou = inter / union

            if target_iou[0] <= iou <= target_iou[1]:
                return mask_x1, mask_x2
            
            gap = 0.0
            if iou < target_iou[0]:
                gap = target_iou[0] - iou
            elif iou > target_iou[1]:
                gap = iou - target_iou[1]

            if gap < best_gap:
                best_gap = gap
                best = (mask_x1, mask_x2, iou)
        # If a suitable IoU is not found after multiple attempts, return the result from the last attempt.

        if best is not None:
            mask_x1, mask_x2, iou = best
            print(f"[WARNING: create_contrastive_views] Fallback: IoU={iou:.3f} (target {target_iou}), tries={max_tries}")
            return mask_x1, mask_x2
    
    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            C, N = self.encoder.num_patches
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)

            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z1 = z # output of encoder for mask contrastive
        z, comb_z = self.predictor(z, mask_x=mask_x)
        # USE_SKIP=True means that only the masked positions use features processed by the predictor, while mask_x uses the original encoded features from the encoder.
        # USE_SKIP=False means that both mask_x and mask_y positions use features processed by the predictor.
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z1, z, r
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad(): 
            mask_x, mask_y = self.make_masks(self.encoder.num_patches)
            h, y = self.forward_target(x, mask_y)
            z1, z, r = self.forward_context(x, mask_x, mask_y)
            loss1 = self.loss_fn(h, z)
            loss2 = self.loss_fn(y, r)

            mask_x1, mask_x2 = self.create_contrastive_views(x, mask_x)
            # z1 = self.encoder(x, self.chans_id.to(x), mask_x=mask_x1)
            z2 = self.encoder(x, self.chans_id.to(x), mask_x=mask_x2)
            loss3 = self.loss_mc(z1, z2, x.shape[0], self.encoder.embed_num)

            act_reg_seq = 0.5 * (z1.pow(2).mean() + z2.pow(2).mean())
            act_reg_weight = 1e-4
            loss = loss2
            if self.USE_LOSS_A:
                loss  = loss + loss1
            if self.USE_LOSS_MC:
                loss  = loss + 0.1 * loss3 + act_reg_weight * act_reg_seq
        
        # -- Contrast
        self.log('valid_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        # -- mc loss
        self.log('valid_mc_loss', loss3, on_epoch=True, on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('valid_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
                
        return loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        # x shape: [64, 58, 1024]
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z1, z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z) # Alightment Constra loss
        loss2 = self.loss_fn(y, r) # Reconstruction loss

        # mc loss
        mask_x1, mask_x2 = self.create_contrastive_views(x, mask_x)
        # z1 = self.encoder(x, self.chans_id.to(x), mask_x=mask_x1)
        z2 = self.encoder(x, self.chans_id.to(x), mask_x=mask_x2)
        loss3 = self.loss_mc(z1, z2, x.shape[0], self.encoder.embed_num)

        act_reg_seq = 0.5 * (z1.pow(2).mean() + z2.pow(2).mean())
        act_reg_weight = 1e-4

        loss = loss2
        if self.USE_LOSS_A:
            loss  = loss + loss1
        if self.USE_LOSS_MC:
            loss  = loss + 0.1 * loss3 + act_reg_weight * act_reg_seq
        
        # -- Contrast
        self.log('train_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('train_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('mc_loss', loss3, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
                
        return loss
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()
        
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)
        
        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res
    
    def configure_optimizers(self):
        
        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)        
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, 
                                                           epochs=max_epochs,
                                                           div_factor = 2,
                                                           final_div_factor=8,
                                                           pct_start = 0.2 ,
                                                           )
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'valid_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        self.wd_scheduler = CosineWDSchedule(
                            optimizer,
                            ref_wd=1e-6,
                            final_wd=1e-6,
                            T_max=int(max_epochs*steps_per_epoch))
        ema = [0.996,1.0]
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
                          for i in range(int(steps_per_epoch*max_epochs)+1))
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        

#-- modeling
def seed_torch(seed=2025):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False
 
