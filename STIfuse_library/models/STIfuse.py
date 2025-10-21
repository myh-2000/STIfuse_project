import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=8, stride=4):
        super().__init__()
        self.seq_len = 16
        self.pred_len = configs.pred_len
        padding = stride

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        self.vlen = configs.vlen
        
        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * 2, self.seq_len)

        #MLP in STIfuse
        self.mlp = nn.Sequential(
            nn.Linear(self.vlen, 4 * self.vlen),
            nn.LayerNorm(4 * self.vlen),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(4 * self.vlen, self.seq_len),
            nn.LayerNorm(self.seq_len)
        )
        
        self.norm4 = nn.LayerNorm(self.seq_len)
        self.input_dim = configs.input_dim

    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        enc_mean = torch.mean(enc_out, dim=1).unsqueeze(1)
        enc_var = torch.var(enc_out, dim=1).unsqueeze(1)
        enc_out = torch.cat([enc_mean, enc_var], dim=1)
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        static_out = self.projection(output)
        static_out = self.norm4(static_out)
        
        # MLP branch
        dynamic_out = self.mlp(x_dec)
        dynamic_out = torch.square(dynamic_out)
        
        return static_out, dynamic_out
    

