import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    


class MultiBranchConv(nn.Module):
    def __init__(self, output_channels=16, attention=True):
        super(MultiBranchConv, self).__init__()
        
        self.branch1 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(1, 16))
        self.branch2 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(2, 16))
        self.branch3 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 16))
        self.branch4 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(4, 16))
        
        self.attn = attention
        self.ca1 = ChannelAttention(output_channels)
        self.ca2 = ChannelAttention(output_channels)
        self.ca3 = ChannelAttention(output_channels)
        self.ca4 = ChannelAttention(output_channels)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)
        self.sa4 = SpatialAttention(kernel_size=7)

    def forward(self, x):

        # Branch 1: No padding needed
        out1 = F.relu(self.branch1(x))

        # Branch 2: Pad to the right (end) so shape becomes (bs, 1, 24, 16)
        x_pad2 = F.pad(x, (0, 0, 0, 1))  # Padding only at the end (on the height dimension)
        out2 = F.relu(self.branch2(x_pad2))

        # Branch 3: Pad one row at the beginning and one at the end (bs, 1, 25, 16)
        x_pad3 = F.pad(x, (0, 0, 1, 1))  # One padding at the start and one at the end
        out3 = F.relu(self.branch3(x_pad3))

        # Branch 4: Pad two rows at the beginning and one at the end (bs, 1, 26, 16)
        x_pad4 = F.pad(x, (0, 0, 1, 2))  # One at the start, Two at the end
        out4 = F.relu(self.branch4(x_pad4))

        # Apply attention if enabled
        if self.attn:
            out1 = out1 * self.ca1(out1)
            out1 = out1 * self.sa1(out1)

            out2 = out2 * self.ca2(out2)
            out2 = out2 * self.sa2(out2)

            out3 = out3 * self.ca3(out3)
            out3 = out3 * self.sa3(out3)

            out4 = out4 * self.ca4(out4)
            out4 = out4 * self.sa4(out4)

        # Remove last dimension of size 1 (from Conv2D)
        out1 = out1.squeeze(-1)
        out2 = out2.squeeze(-1)
        out3 = out3.squeeze(-1)
        out4 = out4.squeeze(-1)

        # Transpose to shape (bs, 23, 16) for concatenation later
        out1 = out1.transpose(1, 2)
        out2 = out2.transpose(1, 2)
        out3 = out3.transpose(1, 2)
        out4 = out4.transpose(1, 2)

        # Concatenate along the last dimension to get shape (bs, 23, 64)
        output = torch.cat((out1, out2, out3, out4), dim=-1)
        
        return output

import torch
import torch.nn as nn
import math

class CRISPRTransformerModel(nn.Module):
    def __init__(self, config):
        super(CRISPRTransformerModel, self).__init__()
        
        # Model parameters
        self.input_dim = 64  # Original input dimension
        self.num_layers = config.get("num_layers", 2)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_prob = config["dropout_prob"]
        self.number_hidden_layers = config["number_hidder_layers"]
        self.seq_length = config.get("seq_length", 20)
        
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_length, self.input_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.input_dim * 4,
            dropout=self.dropout_prob,
            batch_first=True,
            norm_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Convolutional preprocessing (from original model)
        self.conv = MultiBranchConv(attention=config["attn"])
        
        # Hidden layers with residual connections
        self.hidden_layers = []
        start_size = self.seq_length*self.input_dim
        for i in range(self.number_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.GELU(),  # GELU activation (often better than ReLU for transformers)
                nn.Dropout(self.dropout_prob)
            )
            self.hidden_layers.append(layer)
            start_size = start_size // 2
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        
        # Output layer with LayerNorm
        self.output = nn.Linear(start_size, 2)

    def forward(self, x, src_mask=None):
        # Apply conv layer (keeping original preprocessing)
        x = self.conv(x)  # Shape: [batch_size, seq_len, input_dim]
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        x = x.view(x.size(0), -1)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output(x)
        
        return x