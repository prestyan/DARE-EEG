import torch
from Modules.base_model import EEGTransformer
from Modules.prob_model import ConvHead, MLPClassifier, MLPClassifierSmall, MLPClassifierV2
from functools import partial
import os
import torch.nn as nn

class DARE_EEG(nn.Module):
    def __init__(self, 
                 # ConvHead parameter
                 in_channels: int = 22,
                 conv_out_channels: int = 64,
                 in_time_length: int = 1000,
                 conv_out_time_length: int = 1024,
                 conv_layers: int = 3,
                 
                 # Encoder parameter
                 is_eval: bool = False,
                 encoder_path: str = None,
                 complete_model_path: str = None,
                 models_configs = None,
                 freeze_encoder: bool = False,
                 
                 # MLPClassifier parameter
                 N: int = 64,
                 embed_num: int = 1,
                 embed_dim: int = 64,
                 num_classes: int = 3,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.5,
                 pooling_method: str = 'mean'):
        super().__init__()

        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.in_time_length = in_time_length
        self.conv_out_time_length = conv_out_time_length
        self.conv_layers = conv_layers

        self.N = N
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.pooling_method = pooling_method
        self.num_classes = num_classes
        
        # ConvHead: (batch, C, T) -> (batch, conv_out_channels, conv_out_time_length)
        self.conv_head = ConvHead(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            in_time_length=in_time_length,
            out_time_length=conv_out_time_length,
            conv_layers=conv_layers,
            dropout=0.1
        )

        # MLPClassifier: (batch, N, embed_num, embed_dim) -> (batch, num_classes)
        self.classifier = MLPClassifierV2(
            N=N,
            embed_num=embed_num,
            embed_dim=embed_dim,
            num_classes=num_classes,
            # hidden_dims=hidden_dims,
            dropout=dropout,
            pooling_method=pooling_method
        )
        self.freeze_encoder = freeze_encoder
        print('############################################Num classes: ', num_classes)
        
        # Encoder: Load a pre-trained encoder or create a new one.
        if is_eval:
            if not os.path.exists(complete_model_path):
                raise FileNotFoundError(f"[Error] Checkpoint not exist: {complete_model_path}")
            print(f"Load Complete model from {complete_model_path}")
            self.conv_head, self.encoder, self.classifier = self._load_complete_model(complete_model_path, models_configs)
        elif encoder_path and os.path.exists(encoder_path):
            print(f"Loading pretrained encoder from {encoder_path}")
            self.encoder = self._load_pretrained_encoder(encoder_path, models_configs)
        else:
            print("Creating new encoder...")
            self.encoder = self._create_encoder(models_configs)
        
        # 冻结encoder参数
        if freeze_encoder:
            print("❄❄❄❄❄❄❄❄ Freeze Encoder")
            
            # for blk in self.encoder.blocks:
            for p in self.encoder.parameters():
                p.requires_grad = False
    
    def _load_pretrained_encoder(self, encoder_path, models_configs):
        """Loading pre-trained encoder"""
        encoder=EEGTransformer(
            img_size=[58, 256*4],
            patch_size=32*2,
            patch_stride=64,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])
        checkpoint = torch.load(encoder_path, map_location='cpu')
        encoder_stat = {}
        for k,v in checkpoint['state_dict'].items():
            if k.startswith("target_encoder."):
                encoder_stat[k[15:]]=v
        encoder.load_state_dict(encoder_stat)
        return encoder
    
    def _load_complete_model(self, ckpt_path, models_configs):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        conv_head = ConvHead(
            in_channels=self.in_channels,
            out_channels=self.conv_out_channels,
            in_time_length=self.in_time_length,
            out_time_length=self.conv_out_time_length,
            conv_layers=self.conv_layers,
            dropout=self.dropout
        )

        encoder = EEGTransformer(
            img_size=[58, 256*4],
            patch_size=32*2,
            patch_stride=64,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder']
        )

        classifier = MLPClassifier(
            N=self.N,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            pooling_method=self.pooling_method
        )

        conv_head_state = {k[len('conv_head.'):]: v
                           for k, v in state_dict.items()
                           if k.startswith('conv_head.')}
        encoder_state = {k[len('encoder.'):]: v
                         for k, v in state_dict.items()
                         if k.startswith('encoder.')}
        classifier_state = {k[len('classifier.'):]: v
                            for k, v in state_dict.items()
                            if k.startswith('classifier.')}

        missing, unexpected = conv_head.load_state_dict(conv_head_state, strict=False)
        if missing or unexpected:
            print("[Warning] conv_head missing:", missing, "unexpected:", unexpected)

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            print("[Warning] encoder missing:", missing, "unexpected:", unexpected)

        missing, unexpected = classifier.load_state_dict(classifier_state, strict=False)
        if missing or unexpected:
            print("[Warning] classifier missing:", missing, "unexpected:", unexpected)

        print(f"[Info] Loaded complete model from {ckpt_path}")
        return conv_head, encoder, classifier
    
    def _create_encoder(self, models_configs):
        """Create new encoder"""
        encoder=EEGTransformer(
            img_size=[58, 256*4],
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])
        
        return encoder

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(["target_encoder." + x for x in self.encoder.no_weight_decay()])

    def get_num_layers(self):
        return self.encoder.get_num_layers()
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_points)
        Returns:
            logits: (batch, num_classes)
        """
        # ConvHead
        x = self.conv_head(x)  # (batch, conv_out_channels, conv_out_time_length)
        
        # Encoder
        if self.freeze_encoder:
            self.encoder.eval()
        x = self.encoder(x)  # (batch, N, embed_num, embed_dim)

        # MLPClassifier for classification
        logits = self.classifier(x)

        return logits