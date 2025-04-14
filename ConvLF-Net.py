import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(LinearMultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Linear projections for Q, K, and V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Output projection
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.elu = nn.ELU()

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        seq_len, batch_size, d_model = x.size()

        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # Split into multiple heads: (seq_len, batch_size, nhead, d_k)
        Q = Q.view(seq_len, batch_size, self.nhead, self.d_k)
        K = K.view(seq_len, batch_size, self.nhead, self.d_k)
        V = V.view(seq_len, batch_size, self.nhead, self.d_k)

        # Permute dimensions to (batch_size, nhead, seq_len, d_k)
        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        # Apply non-linearity to ensure all values are positive
        Q = self.elu(Q) + 1
        K = self.elu(K) + 1

        # Compute key-value product: shape (batch_size, nhead, d_k, d_k)
        KV = torch.matmul(K.transpose(-2, -1), V)

        # Compute attention output: shape (batch_size, nhead, seq_len, d_k)
        out = torch.matmul(Q, KV)

        # Reshape back to (seq_len, batch_size, d_model)
        out = out.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, d_model)
        out = self.dropout(self.linear_out(out))
        return out


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.linear_multihead_attention = LinearMultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Attention sub-layer
        src2 = self.linear_multihead_attention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward sub-layer
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoderModule, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
             for _ in range(num_encoder_layers)]
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class Model_linear(nn.Module):
    def __init__(self, channel_in=13,num_classes,num_filters=16, d_model=256, nhead=8, num_encoder_layers=2, dim_feedforward=1024,
                 dropout=0.2):
        super(Model_linear, self).__init__()
        # Multi-scale CNN branches
        self.cnn1 = nn.Sequential(
            nn.Conv1d(channel_in, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(channel_in, 64, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
            nn.Conv1d(64, 64, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(channel_in, 64, kernel_size=50, padding=25),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
            nn.Conv1d(64, 64, kernel_size=50, padding=25),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(channel_in, 64, kernel_size=100, padding=50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
            nn.Conv1d(64, 64, kernel_size=100, padding=50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2),
        )

        # Projection to transformer dimension
        self.flatten_and_project = nn.Linear(64, d_model)

        # Transformer encoder
        self.linear_transformer_encoder = TransformerEncoderModule(d_model, nhead, num_encoder_layers, dim_feedforward,
                                                                   dropout)

        # Final classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x0 = x.permute(0, 2, 1)  # (batch_size, channels, sequence_length)
        x1 = self.cnn1(x0)
        x2 = self.cnn2(x0)
        x3 = self.cnn3(x0)
        x4 = self.cnn4(x0)

        # Fuse features from all CNN branches
        x = x1 + x2 + x3 + x4

        x = x.permute(2, 0, 1)  # (sequence_length, batch_size, channels)
        x = self.flatten_and_project(x)  # (sequence_length, batch_size, d_model)
        x = self.linear_transformer_encoder(x)  # (sequence_length, batch_size, d_model)
        x = x.mean(dim=0)  # Average over sequence dimension -> (batch_size, d_model)
        x = self.classifier(x)  # (batch_size, num_classes)
        return x
