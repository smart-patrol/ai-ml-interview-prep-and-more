import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Union


seed = 172
torch.manual_seed(seed)

# Let's now see that in action. In the exercise below, you will build an attention module in Pytorch. Specifically, you will develop the "general" scoring function using softmax. The module will receive the vectors y and h and calculate using softmax the attention weights.
# you will need to transpose the y vector for this calculation.
# Softmax( y^T * W * h )

class Attention(nn.Module):

    def __init__(self, y_dim: int, h_dim: int):
        super().__init__()
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.W = nn.Linear(y_dim, h_dim, bias=False)


    def forward(self,y: torch.Tensor, h: torch.Tensor):
        """
        :param y: (batch_size, y_dim)
        :param h: (batch_size, h_dim)
        :return: (batch_size, h_dim)
        """
        y = y.transpose(1,0)
        y = self.W(y)
        y = y.transpose(1,0)
        y = torch.matmul(y, h.transpose(1,0))
        y = F.softmax(y, dim=1)
        y = torch.matmul(y, h)
        return y


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, 
                query: torch.FloatTensor, 
                key: torch.FloatTensor,
                value: torch.FloatTensor, 
                mask: Optional[torch.ByteTensor] = None, 
                dropout: Optional[nn.Dropout] = None
                ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)
            `dropout`: nn.Dropout
        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1) 
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
