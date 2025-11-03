import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.init import xavier_normal_
from einops import repeat, rearrange


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size=3, stride=1):
        super(moving_avg, self).__init__()
        
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
    
class EmbeddingSar(nn.Module):
    """
    Embedding the SAR trend to feature space
    """

    def __init__(self, c_in, d_model, kernel_size=3, stride=1, center_weight=0.5):
        super(EmbeddingSar, self).__init__()
        self.window_size = kernel_size
        self.center_weight = center_weight
        self.moving_avg = moving_avg(kernel_size, stride)
        self.linear = nn.Linear(c_in, d_model, bias=False)
        # self.conv1D = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                         kernel_size=3, padding=1, stride=1)
     
    def forward(self, x, timestamps):
        # rvi = (4. * x[..., 1]) / (x[..., 0] + x[..., 1])
        # x = torch.concat([x, rvi.unsqueeze(-1)], dim=-1)
        # conv1D is worser
        # x = self.conv1D(x.permute(0, 2, 1)).permute(0, 2, 1)
        # if not self.is_regular_intervals(timestamps):
        #     x, timestamps = self.weighted_moving_avg(x, timestamps)
        # else:
        #     x = self.moving_avg(x)
        x = self.linear(x)
        return x, timestamps
    
    def is_regular_intervals(self, timestamps):
        intervals = torch.diff(timestamps)
        return torch.allclose(intervals, intervals[0])

    def weighted_moving_avg(self, x, timestamps):
        _, T, _ = x.shape
        half_window = self.window_size // 2
        smoothed = torch.zeros_like(x)
        timestamps_new = timestamps.clone()

        for i in range(T):
            if (i < half_window) or (i >= T - half_window):
                start = max(0, i - 1)
                end = min(T, i + 2)
                window = x[:, start:end, :]
                time_window = timestamps[start:end]

                # Calculate weights based on time differences
                center_time = time_window.mean().long()
                time_diffs = torch.abs(time_window - center_time)
                weights = 1 / (time_diffs + 1e-6)  # Add small value to avoid division by zero
                weights = weights / weights.sum()  # Normalize weights

                smoothed[:, i, :] = (window * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                timestamps_new[i] = time_window.mean().long()
            else:
                start = max(0, i - half_window)
                end = min(T, i + half_window + 1)
                window = x[:, start:end, :]
                time_window = timestamps[start:end]

                # Calculate weights based on time differences
                center_time = timestamps[i]
                center_idx = (end - start) // 2
                time_diffs = torch.abs(time_window - center_time)
                weights = 1 / (time_diffs + 1e-6)  # Add small value to avoid division by zero
                
                weights[center_idx] = self.center_weight # Set the center weight to 0.5 and normalize the rest
                weights = weights / (weights.sum() - self.center_weight) * (1 - self.center_weight) # Normalize weights
                weights[center_idx] = self.center_weight
                
                smoothed[:, i, :] = (window * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        return smoothed, timestamps_new
    

class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # q * K, k where equal zeros fill -1e9 in the col of atten
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    """position-wise feed-forward network of the Transformer block"""
    
    def __init__(self, d_in, d_hid, dropout=0.1, with_res=True):
        super().__init__()
        self.with_res = with_res
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # x = self.layer_norm(x) # delete
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        if self.with_res:
            x += residual
        x = self.layer_norm(x) 
        return x


class EncoderLayer(nn.Module):
    """Stacked Transformer blocks"""
    
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = "cuda"
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, attn_mask=None):
        if self.diagonal_attention_mask:
            diag_attn_mask = torch.eye(self.d_time).to(self.device) # d_t * d_t
            diag_attn_mask = diag_attn_mask.unsqueeze(0).repeat(enc_input.size(0), 1, 1) # → bs * d_t * d_t
        else:
            diag_attn_mask = None
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attn_mask.size(-1), 1)  # bs * d_t → bs * d_t * d_t
            attn_mask = attn_mask + diag_attn_mask
            attn_mask = torch.where(attn_mask > 1, 1, attn_mask)
        else:   
            attn_mask = diag_attn_mask
        
        residual = enc_input
        
        # whether apply LN before attention cal, namely Pre-LN,
        # refer paper https://arxiv.org/abs/2002.04745
        # enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=attn_mask
        )
        enc_output = residual + self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class MultiModalEncoderLayer(nn.Module):
    """Transformer block of the Low-rank Fusion module"""
    
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True
    ):
        super(MultiModalEncoderLayer, self).__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = "cuda"
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, attn_mask=None): 
        attn_mask = None
        
        residual = enc_input
        
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=attn_mask
        )
        enc_output = residual + self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights
    

class DecoderLayer(nn.Module):
    """Time-aware Decoder"""
    
    def __init__(
        self,
        d_model,
        n_head,
        d_k,
        d_v,
        attn_dropout=0.1,
    ):
        super(DecoderLayer, self).__init__()
        
        self.timeindex_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)

    def forward(self, dates, dec_input):
        dec_output, attn_weights = self.timeindex_attn(
            dates, dec_input, dec_input
        )
        return dec_output, attn_weights


class TensorFusion(nn.Module):
    """Vanilla Tensor Fusion"""
    
    def __init__(self, d_time, d_model):
        super(TensorFusion, self).__init__()
        self.d_time = d_time
        self.post_fusion = nn.Linear((d_model+1) ** 2, d_model)
        
    def forward(self, X, X_aux):
        b, t, _ = X.shape
        X = torch.cat([torch.ones(size=(b, t, 1)).to(X.device), X], dim=2)
        X_aux = torch.cat([torch.ones(size=(b, t, 1)).to(X_aux.device), X_aux], dim=2)
        
        tf = torch.matmul(X.unsqueeze(3), X_aux.unsqueeze(2))
        output = self.post_fusion(tf.view(tf.shape[0], self.d_time, -1))
        return output
    
        
class LowRankTensorFusion(nn.Module):
    """Efficient Low-rank Multimodal Fusion with Modality-Specific Factors"""
    
    def __init__(self, d_model, rank=8):
        super(LowRankTensorFusion, self).__init__()
        self.w_x = nn.Parameter(torch.Tensor(rank, d_model + 1, d_model))
        self.w_x_aux = nn.Parameter(torch.Tensor(rank, d_model + 1, d_model))
        
        xavier_normal_(self.w_x)
        xavier_normal_(self.w_x_aux)
        
    def forward(self, X, X_aux):
        b, t, _ = X.shape
        X = torch.cat([torch.ones(size=(b, t, 1)).to(X.device), X], dim=2)
        X_aux = torch.cat([torch.ones(size=(b, t, 1)).to(X_aux.device), X_aux], dim=2)
        
        X = rearrange(X, 'b t m -> (b t) m')
        X_aux = rearrange(X_aux, 'b t m -> (b t) m')
        
        output = torch.matmul(X, self.w_x) * torch.matmul(X_aux, self.w_x_aux) 
        output = torch.sum(output, dim=0).squeeze()
        output = rearrange(output, '(b t) m -> b t m', b=b)
        return output
        
    
class PositionalEncoding(nn.Module):
    """Time (DOY) embedding layer"""
    
    def __init__(self, d_hid, n_position=366, T=365):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid, T)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid, T):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(T, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).cuda() # n_position * d_hid

    def forward(self, x):
        # return self.pos_table.clone().detach() # 1 367 256
        return self.pos_table[x, :].clone().detach()
    

if __name__ == "__main__":

    batch_size = 128
    T = 10
    Bands = 2

    x = torch.randn(batch_size, T, Bands)
    # timestamps = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=torch.float32)
    timestamps = torch.tensor([0, 5, 12, 17, 20, 25, 30, 35, 40, 45], dtype=torch.float32)

    c_in = Bands
    d_model = 256
    window_size = 3
    model = EmbeddingSar(c_in, d_model, window_size)
    output, new_timestamps = model(x, timestamps)

    print("Input x:", x)
    print("Input timestamps:", timestamps)
    print("Output x:", output)
    print("New timestamps:", new_timestamps)