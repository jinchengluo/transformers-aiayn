import torch

from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention
from transformer import PositionwiseFeedForwardNetwork

"""
Unit tests for ScaledDotProductAttention

Checks if the output shape matches the expected dimensions
Checks if the weights sum to 1 across the last dimension
"""
batch_size, seq_len, d_k = 2, 4, 8
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

attention = ScaledDotProductAttention(d_k, d_k)
output, weights = attention(Q, K, V)

assert output.shape == (batch_size, seq_len, d_k)
assert weights.shape == (batch_size, seq_len, seq_len)
assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len))

print("ScaledDotProductAttention passed all checks.")

"""
Unit tests for MultiHeadAttention

Checks if the model dimension is divisible by the number of heads
Checks if the output shape matches the expected dimensions
"""
batch_size, seq_len, d_model, nb_heads = 2, 4, 16, 4
assert d_model % nb_heads == 0

multihead = MultiHeadAttention(d_model, nb_heads)
x = torch.randn(batch_size, seq_len, d_model)
out = multihead(x, x, x)

assert out.shape == (batch_size, seq_len, d_model)

print("MultiHeadAttention passed all checks.")

"""
Unit tests for PositionwiseFeedForwardNetwork

Checks if the output shape matches the expected dimensions
Checks if the gradients are not NaN
"""
batch_size, seq_len, model_dim, inner_dim = 2, 5, 16, 64

ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
x = torch.randn(batch_size, seq_len, model_dim)
output = ffn(x)

assert output.shape == (batch_size, seq_len, model_dim), f"Bad shape: {output.shape}"

output.sum().backward()
for name, param in ffn.named_parameters():
    if param.grad is not None:
        assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"

print("PositionwiseFeedForwardNetwork passed all checks.")