import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    # compute attention score by matrix multiplication Q and K
    matmul_qk = torch.bmm(q, k.transpose(1, 2).contiguous())

    # scale matmul_qk
    dk = k.size(-1)
    scaled_attention_score = matmul_qk / math.sqrt(dk)

    if mask is not None:
        scaled_attention_score = scaled_attention_score.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scaled_attention_score, dim=-1)
    output = torch.bmm(attention_weights, v)
    return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dim):
        super(SelfAttention, self).__init__()
        self.wq = nn.Linear(hidden_size, dim)
        self.wk = nn.Linear(hidden_size, dim)
        self.wv = nn.Linear(hidden_size, dim)

    def forward(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        output, attention_weights = scaled_dot_product_attention(q, k, v)
        return output, attention_weights

