import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_dimension, value_dimension):
        super().__init__()
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension

    def forward(self, queries, keys, values, mask=None):
        dot_product = torch.matmul(queries, keys.transpose(-2, -1))
        dot_product /= math.sqrt(self.key_dimension)
        if mask is not None:
            dot_product.masked_fill_(mask == 0, float("-inf"))
        weights = dot_product.softmax(dim=-1)
        return torch.matmul(weights, values), weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.key_dimension = model_dimension // number_of_heads
        self.value_dimension = model_dimension // number_of_heads
        self.linear_queries = nn.Linear(model_dimension, model_dimension) # W_Q
        self.linear_keys = nn.Linear(model_dimension, model_dimension) # W_K
        self.linear_values = nn.Linear(model_dimension, model_dimension) # W_V
        self.linear_output = nn.Linear(model_dimension, model_dimension) # w_0

    def forward(self, queries, keys, values, mask=None):         
        projected_queries = self.linear_queries(queries) # Q * W_Q
        projected_queries = projected_queries.view(projected_queries.shape[0], projected_queries.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_keys = self.linear_keys(keys) # K * W_K
        projected_keys = projected_keys.view(projected_keys.shape[0], projected_keys.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_values = self.linear_values(values) # V * W_V
        projected_values = projected_values.view(projected_values.shape[0], projected_values.shape[1], self.number_of_heads, self.value_dimension).transpose(1, 2)
    
        scaled_dot_product = ScaledDotProductAttention(self.key_dimension, self.value_dimension)
        attention, _ = scaled_dot_product(queries, keys, values, mask)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(attention.shape[0], -1, self.model_dimension)

        return self.linear_output(attention)


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimension, inner_layer_dimension):
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, inner_layer_dimension) # W1 and b1
        self.linear2 = nn.Linear(inner_layer_dimension, model_dimension) # W2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        return self.linear2(output)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, model_dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.model_dimension = model_dimension
        self.embedding = nn.Embedding(vocabulary_size, model_dimension)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dimension)
    

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.model_dimension = model_dimension
        
        positional_encoding = torch.zeros(sequence_length, model_dimension)

        positions = torch.arange(0, sequence_length, 1, dtype=float)
        positions = torch.unsqueeze(positions, 1) 
        denominator = torch.exp(torch.arange(0, model_dimension, 2, dtype=float) * -math.log(10000.) / model_dimension)

        positional_encoding[:,0::2] = torch.sin(positions * denominator)
        positional_encoding[:,1::2] = torch.sin(positions * denominator)

        self.register_buffer("positional_encoding", positional_encoding)       

    def forward(self, x_embedded):
        return x_embedded + self.positional_encoding[:x_embedded.shape[1]]
    
