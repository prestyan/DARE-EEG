import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Important: If the dataset sampling rate is similar to that of the pre-trained model, 
using sampling rate adaptation is not recommended (this has been commented out).
'''

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.doWeightNorm = doWeightNorm
        self.max_norm = max_norm

    def forward(self, x):
        if self.doWeightNorm:
            with torch.no_grad():
                # Apply L2 max-norm (dim=0) to the weight vector of each output neuron.
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class ConvHead(nn.Module):
    """
    Project (B, C, T) to (B, C1, T1).
    - If intermediate length T' >= T1: Compress to T1 using AdaptiveAvgPool1d
    - If T' < T1: Upsample to T1 using linear interpolation (or use ConvTranspose1d instead)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_time_length: int,
                 out_time_length: int,
                 conv_layers: int = 3,
                 kernel_size: int = 15,
                 dropout: float = 0.2):
        super().__init__()
        self.out_time_length = out_time_length

        layers = []
        cur_c = in_channels
        # adapt channels
        for i in range(conv_layers):
            nxt_c = out_channels if i == conv_layers - 1 else min(cur_c * 2, 256)
            layers += [
                Conv1dWithConstraint(cur_c, nxt_c,
                          kernel_size=1,
                          # padding=kernel_size // 2,
                          bias=False,
                          max_norm=1),
                nn.BatchNorm1d(nxt_c),
                nn.GELU(),
            ]
            cur_c = nxt_c
        # adapt time space (sampling)
        layers += [Conv1dWithConstraint(out_channels, out_channels,
                          kernel_size=15,
                          # padding=kernel_size // 2,
                          bias=False, groups=out_channels, max_norm=1),
                   nn.BatchNorm1d(out_channels),
                   nn.GELU()]
        layers += [nn.Dropout1d(dropout)]
        self.conv_layers = nn.Sequential(*layers)
        self.anti_alias = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels,
                                    bias=False)

    def forward(self, x):  # x: (B, C, T)
        x = self.conv_layers(x)  # -> (B, C1, T')
        T1 = self.out_time_length
        Tprime = x.size(-1)
        return x
        '''
        if Tprime == T1:
            return x
        elif Tprime > T1:
            # Downsampling to T1 (adaptive average pooling, numerical stability, anti-aliasing)
            # Lightweight deep separable lowpass
            x = self.anti_alias(x)
            return F.adaptive_avg_pool1d(x, T1)
        else:
            # Upsample to T1 (linear interpolation); 
            # can also be changed to mode="nearest"/"cubic" or use ConvTranspose1d learning upsampling
            return F.interpolate(x, size=T1, mode="linear")
        '''


class MLPClassifier(nn.Module):
    """
    MLP projection layer: multi-classify the input (batch, N, embed_num, embed_dim)
    """

    def __init__(self,
                 N: int,  # seq length
                 embed_num: int,  # summary tokens embed num
                 embed_dim: int,  # transformer embed dimension
                 num_classes: int,  # Number of classification categories
                 hidden_dims: list = [256, 128],  # Hidden layer dimensions
                 dropout: float = 0.1,
                 pooling_method: str = 'mean',  # 'mean', 'max', 'cls', 'attention'
                 use_cls_token: bool = True):
        super().__init__()

        assert pooling_method in ['mean', 'max', 'cls', 'attention']
        self.N = N
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_method = pooling_method
        self.use_cls_token = use_cls_token

        # Calculate input feature dimensions
        if pooling_method == 'cls' and use_cls_token:
            # If using CLS token, use all
            input_dim = N * embed_dim * embed_num
        elif pooling_method == 'attention':
            # If attention pooling is used, an additional attention layer is required
            input_dim = embed_dim
            self.attention_pool = AttentionPooling(embed_dim)
        else:
            # Other cases: dimensions after mean/max pooling
            input_dim = N * embed_num * embed_dim

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                LinearWithConstraint(current_dim, hidden_dim, max_norm=1),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(LinearWithConstraint(current_dim, num_classes, max_norm=0.25))

        self.mlp = nn.Sequential(*layers)

    def pool_features(self, x):
        """
        Perform pooling operations on features
        Args:
            x: (batch, N, embed_num, embed_dim)
        Returns:
            pooled: (batch, feature_dim)
        """
        batch_size = x.size(0)

        if self.pooling_method == 'cls' and self.use_cls_token:
            # use CLS token
            x_flat = x.flatten(1, 3)
            return x_flat  # (batch, embed_dim)

        elif self.pooling_method == 'mean':
            # Global average pooling
            return x.flatten(1, 3)
            return x.mean(dim=1).flatten(1)  # (batch, embed_num * embed_dim)

        elif self.pooling_method == 'max':
            # Global max pooling
            return x.max(dim=1)[0].flatten(1)  # (batch, embed_num * embed_dim)

        elif self.pooling_method == 'attention':
            # Attention pooling
            x_flat = x.view(batch_size, -1, self.embed_dim)  # (batch, N*embed_num, embed_dim)
            return self.attention_pool(x_flat)  # (batch, embed_dim)

        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def forward(self, x):
        """
        Args:
            x: (batch, N, embed_num, embed_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # pooling features
        pooled_features = self.pool_features(x)
        # MLP classification
        logits = self.mlp(pooled_features)

        return logits
    
class MLPClassifierSmall(nn.Module):
    """
    MLP projection layer: multi-classify the input (batch, N, embed_num, embed_dim)
    """

    def __init__(self,
                 N: int,  # seq length
                 embed_num: int,  # summary tokens embed num
                 embed_dim: int,  # transformer embed dimension
                 num_classes: int,  # Number of classification categories
                 hidden_dims: list = [256, 128],  # Hidden layer dimensions
                 dropout: float = 0.1,
                 pooling_method: str = 'mean',  # 'mean', 'max', 'cls', 'attention'
                 use_cls_token: bool = True):
        super().__init__()

        assert pooling_method in ['mean', 'max', 'cls', 'attention']
        self.N = N
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_method = pooling_method
        self.use_cls_token = use_cls_token

        # Calculate input feature dimensions
        if pooling_method == 'cls' and use_cls_token:
            # If using CLS token, use all
            input_dim = N * embed_dim * embed_num
        elif pooling_method == 'attention':
            # If attention pooling is used, an additional attention layer is required
            input_dim = embed_dim
            self.attention_pool = AttentionPooling(embed_dim)
        else:
            # Other cases: dimensions after mean/max pooling
            input_dim = N * embed_num * embed_dim

        layers = []
        current_dim = input_dim

        layers.append(nn.Dropout(dropout))

        layers.append(LinearWithConstraint(current_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def pool_features(self, x):
        """
        Perform pooling operations on features
        Args:
            x: (batch, N, embed_num, embed_dim)
        Returns:
            pooled: (batch, feature_dim)
        """
        batch_size = x.size(0)

        if self.pooling_method == 'cls' and self.use_cls_token:
            # use CLS token
            x_flat = x.flatten(1, 3)
            return x_flat  # (batch, embed_dim)

        elif self.pooling_method == 'mean':
            # Global average pooling
            return x.flatten(1, 3)
            return x.mean(dim=1).flatten(1)  # (batch, embed_num * embed_dim)

        elif self.pooling_method == 'max':
            # Global max pooling
            return x.max(dim=1)[0].flatten(1)  # (batch, embed_num * embed_dim)

        elif self.pooling_method == 'attention':
            # Attention pooling
            x_flat = x.view(batch_size, -1, self.embed_dim)  # (batch, N*embed_num, embed_dim)
            return self.attention_pool(x_flat)  # (batch, embed_dim)

        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def forward(self, x):
        """
        Args:
            x: (batch, N, embed_num, embed_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # pooling features
        pooled_features = self.pool_features(x)
        # MLP classification
        logits = self.mlp(pooled_features)

        return logits
    
class MLPClassifierV2(nn.Module):
    """
    MLP projection layer: multi-classify the input (batch, N, embed_num, embed_dim)
    """

    def __init__(self,
                 N: int,  # seq length
                 embed_num: int,  # summary tokens embed num
                 embed_dim: int,  # transformer embed dimension
                 num_classes: int,  # Number of classification categories
                 hidden_dims: list = [128, 16],  # Hidden layer dimensions
                 dropout: float = 0.1,
                 pooling_method: str = 'mean',  # 'mean', 'max', 'cls', 'attention'
                 use_cls_token: bool = True):
        super().__init__()

        assert pooling_method in ['mean', 'max', 'cls', 'attention']
        self.N = N
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_method = pooling_method
        self.use_cls_token = use_cls_token

        input_dim = embed_num * embed_dim

        layers = []
        current_dim = input_dim
        dp = [dropout, dropout - 0.2]
        i = 0

        for hidden_dim in hidden_dims:
            layers.extend([
                LinearWithConstraint(current_dim, hidden_dim, max_norm=1),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dp[i])
            ])
            current_dim = hidden_dim
            i = 1 - i

        # layers.append(LinearWithConstraint(current_dim, num_classes, max_norm=0.25))
        self.layer2 = LinearWithConstraint(N * current_dim, num_classes, max_norm=0.25)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, N, embed_num, embed_dim)
        Returns:
            logits: (batch, num_classes)
        """
        x = x.flatten(2)  # (batch, embed_num, embed_dim)
        # MLP classification
        x = self.mlp(x)
        x = x.flatten(1)
        logits = self.layer2(x)

        return logits


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            pooled: (batch, embed_dim)
        """

        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, embed_dim)

        return pooled
