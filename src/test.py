import torch

from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention

batch_size, seq_len, d_k = 2, 4, 8
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

# Unit tests for ScaledDotProductAttention

attention = ScaledDotProductAttention(d_k, d_k)
output, weights = attention(Q, K, V)

assert output.shape == (batch_size, seq_len, d_k)
assert weights.shape == (batch_size, seq_len, seq_len)
assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

print("ScaledDotProductAttention passed all checks.")

# Unit tests for MultiHeadAttention

batch_size, seq_len, d_model, nb_heads = 2, 4, 16, 4
assert d_model % nb_heads == 0

multihead = MultiHeadAttention(d_model, nb_heads)
x = torch.randn(batch_size, seq_len, d_model)
out = multihead(x, x, x)

assert out.shape == (batch_size, seq_len, d_model)

print("MultiHeadAttention passed all checks.")