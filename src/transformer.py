import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_dimension, value_dimension):
        super().__init__
        self.keys_dimension = key_dimension
        self.value_dimension = value_dimension

    def forward(self, queries, keys, values, mask=None):
        dot_product = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.key_dimension)
        if mask is not None:
            dot_product.masked_fill_(mask == 0, float("-inf"))
        weights = nn.Softmax(dot_product)
        return torch.matmul(weights, values)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads):
        super().__init__
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.key_dimension = model_dimension // number_of_heads
        self.value_dimension = model_dimension // number_of_heads
        self.linear_queries = nn.Linear(model_dimension, model_dimension) # W_Q
        self.linear_keys = nn.Linear(model_dimension, model_dimension) # W_K
        self.linear_values = nn.Linear(model_dimension, model_dimension) # W_V
        self.linear_output = nn.Linear(model_dimension, model_dimension) # w_0

    def forward(self, queries, keys, values, mask):
        head = torch.asarray([])
        
        projected_queries = self.linear_queries(queries) # Q * W_Q
        projected_queries = projected_queries.view(projected_queries.shape[0], projected_queries.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_keys = self.linear_keys(keys) # K * W_K
        projected_keys = projected_keys.view(projected_keys.shape[0], projected_keys.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_values = self.linear_values(values) # V * W_V
        projected_values = projected_values.view(projected_values.shape[0], projected_values.shape[1], self.number_of_heads, self.value_dimension).transpose(1, 2)
    
        scaled_dot_product = ScaledDotProductAttention(projected_queries, projected_keys, projected_values, self.key_dimension, self.value_dimension)
        attention = scaled_dot_product(queries, keys, values, mask)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(attention.shape[0], -1, self.model_dimension)

        return self.linear_output(attention)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, model_dimension):
        super().__init__
        self.model_dimension = model_dimension
        self.vocab_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, model_dimension)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dimension)