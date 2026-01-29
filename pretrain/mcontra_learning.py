import torch
from torch import nn
import torch.nn.functional as F

class MaskContrastiveLearning(nn.Module):
    def __init__(self, models_configs, use_cls: bool = True, cls_position: str = 'end',  # 'front' or 'end'
                 learnable_temp: bool = False):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(models_configs['encoder']['embed_dim'] * models_configs['encoder']['embed_num'], 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Contrastive learning loss function
        # self.contrastive_loss_fn = nn.CosineSimilarity(dim=1)
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('temperature', torch.tensor(0.1))
        
        assert cls_position in ('front', 'end')
        self.use_cls = use_cls
        self.cls_position = cls_position
    
    @staticmethod
    def _masked_avg_pool(z: torch.Tensor, attn_mask: torch.Tensor | None):
    # In our case, `attn_mask` is usually `None`.
    # When there is no padding (for example, EEG data always has a fixed length of 512), `attn_mask` is actually completely unnecessary.

        if attn_mask is None:
            return z.mean(dim=1)
        denom = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (z * attn_mask.unsqueeze(-1)).sum(dim=1) / denom

    def _pool(self, z: torch.Tensor, embed_num, attn_mask: torch.Tensor | None):
        # We fix the code so people can jump this step now.
        if self.use_cls:
            # When `use_cls=True`, we directly use `z[:,0]` (the CLS token) for comparison, avoiding the influence of padding on the mean.
            if self.cls_position == 'front':
                cls_tokens = z[:, :, :embed_num, :]   
            else:
                cls_tokens = z[:, :, -embed_num: , :]  
            cls_tokens = cls_tokens.mean(dim=1)
            cls_tokens = cls_tokens.flatten(1, 2)  # (B, embed_num*D)
            return cls_tokens.squeeze(1) 
        
        return self._masked_avg_pool(z, attn_mask)
    
    def compute_contrastive_loss(self, z1, z2, batch_size, embed_num,
                                 attn_mask1: torch.Tensor | None = None,
                                 attn_mask2: torch.Tensor | None = None):

        h1 = self._pool(z1, embed_num, attn_mask1)                  # [B, D]
        h2 = self._pool(z2, embed_num, attn_mask2)                  # [B, D]
        z1_proj = self.projection_head(h1)  
        z2_proj = self.projection_head(h2)
        
        z1_proj = F.normalize(z1_proj, dim=1)
        z2_proj = F.normalize(z2_proj, dim=1)
        
        sim_matrix = torch.matmul(z1_proj, z2_proj.T) / self.temperature.clamp(min=1e-6)
        
        labels = torch.arange(batch_size).to(z1.device)
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_1 + loss_2) * 0.5
    
    def forward(self, z1, z2, batch_size, embed_num,  
                attn_mask1: torch.Tensor | None = None,
                attn_mask2: torch.Tensor | None = None):
        return self.compute_contrastive_loss(z1, z2, batch_size, embed_num, attn_mask1, attn_mask2)
