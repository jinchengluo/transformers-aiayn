import copy
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
            if mask.dim() == 4 and mask.size(-2) == 1:
                mask = mask.expand(-1, -1, dot_product.size(-2), -1)
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
        attention, _ = scaled_dot_product(projected_queries, projected_keys, projected_values, mask)
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
        output_ids = self.linear1(x)
        output_ids = self.relu(output_ids)
        return self.linear2(output_ids)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, model_dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.model_dimension = model_dimension
        self.embedding = nn.Embedding(vocabulary_size, model_dimension)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dimension)
    

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, max_sequence_length=5000):
        super().__init__()
        self.sequence_length = max_sequence_length
        self.model_dimension = model_dimension
        
        positional_encoding = torch.zeros(max_sequence_length, model_dimension)

        positions = torch.arange(0, max_sequence_length, 1, dtype=float)
        positions = torch.unsqueeze(positions, 1) 
        denominator = torch.exp(torch.arange(0, model_dimension, 2, dtype=float) * -math.log(10000.) / model_dimension)

        positional_encoding[:,0::2] = torch.sin(positions * denominator)
        positional_encoding[:,1::2] = torch.cos(positions * denominator)

        self.register_buffer("positional_encoding", positional_encoding)       

    def forward(self, x_embedded):
        return x_embedded + self.positional_encoding[:x_embedded.shape[1]]


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, multihead_attention, feedforward_network):
        super().__init__()
        self.model_dimension = model_dimension
        self.multihead_attention = multihead_attention
        self.feedforward_network = feedforward_network
        self.layernorm1 = nn.LayerNorm(model_dimension)
        self.layernorm2 = nn.LayerNorm(model_dimension)

    def forward(self, input_embedded):
        attention_ouput = self.multihead_attention(input_embedded, input_embedded, input_embedded)
        sublayer_output1 = self.layernorm1(input_embedded + attention_ouput)
        ffnetwork_output = self.feedforward_network(sublayer_output1)
        sublayer_output2 = self.layernorm2(sublayer_output1 + ffnetwork_output)
        return sublayer_output2
    
class Encoder(nn.Module):
    def __init__(self, model_dimension, number_of_layers, encoder_layer):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(number_of_layers)])

    def forward(self, input_embedded):
        input_encode = input_embedded
        for layer in self.layers:
            input_encode = layer(input_encode)
        return input_encode
    

class DecoderLayer(nn.Module):
    def __init__(self, model_dimension, masked_multihead_attention, multihead_attention, feedforward_network):
        super().__init__()
        self.model_dimension = model_dimension
        self.masked_multihead_attention = masked_multihead_attention
        self.multihead_attention = multihead_attention
        self.feedforward_network = feedforward_network
        self.layernorm1 = nn.LayerNorm(model_dimension)
        self.layernorm2 = nn.LayerNorm(model_dimension)
        self.layernorm3 = nn.LayerNorm(model_dimension)

    def forward(self, input_encode, output_embedded, mask):
        masked_attention_output = self.masked_multihead_attention(queries=output_embedded, keys=output_embedded, values=output_embedded, mask=mask)
        sublayer_output1 = self.layernorm1(output_embedded + masked_attention_output)
        attention_ouput = self.multihead_attention(queries=sublayer_output1, keys=input_encode, values=input_encode)
        sublayer_output2 = self.layernorm2(sublayer_output1 + attention_ouput)
        ffnetwork_ouput = self.feedforward_network(sublayer_output2)
        sublayer_output3 = self.layernorm3(sublayer_output2 + ffnetwork_ouput)
        return sublayer_output3
    

class Decoder(nn.Module):
    def __init__(self, model_dimension, number_of_layers, decoder_layer):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(number_of_layers)])

    def forward(self, input_encode, output_embedded, mask):
        output_encode = output_embedded
        for layer in self.layers:
            output_encode = layer(input_encode, output_encode, mask)
        return output_encode


class Transformer(nn.Module):
    def __init__(self, model_dimension, inner_layer_dimension, number_of_layers, number_of_heads, input_vocabulary_size, output_vocabulary_size):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.input_vocabulary_size = input_vocabulary_size
        self.output_vocabulary_size = output_vocabulary_size

        self.input_embedding = Embedding(input_vocabulary_size, model_dimension)
        self.output_embedding = Embedding(output_vocabulary_size, model_dimension)

        self.input_pos_encoding = PositionalEncoding(model_dimension)
        self.output_pos_encoding = PositionalEncoding(model_dimension)

        masked_multihead_attention = MultiHeadAttention(model_dimension, number_of_heads)
        multihead_attention = MultiHeadAttention(model_dimension, number_of_heads)
        feedforward_network = PositionwiseFeedForwardNetwork(model_dimension, inner_layer_dimension)
        encoder_layer = EncoderLayer(model_dimension, multihead_attention, feedforward_network)
        decoder_layer = DecoderLayer(model_dimension, masked_multihead_attention, multihead_attention, feedforward_network)

        self.encoder = Encoder(model_dimension, number_of_layers, encoder_layer)
        self.decoder = Decoder(model_dimension, number_of_layers, decoder_layer)

        self.linear_projection = nn.Linear(model_dimension, output_vocabulary_size)

    def forward(self, input_ids, output_ids, mask):
        input_embedded = self.input_embedding(input_ids)
        input_pos_encoded = self.input_pos_encoding(input_embedded)
        input_encoded = self.encoder(input_pos_encoded)

        output_embedded = self.output_embedding(output_ids)
        output_pos_encoded = self.output_pos_encoding(output_embedded)
        output_decoded = self.decoder(input_encoded, output_pos_encoded, mask)

        output_decoded = self.linear_projection(output_decoded)
        output_decoded = output_decoded.softmax(dim=-1)

        return output_decoded


def create_causal_mask(seq_len):
    """
    Crée un masque causal (triangulaire inférieur) pour l'auto-attention masquée
    Empêche de voir les tokens futurs
    Retourne: (seq_len, seq_len) - 1 pour positions autorisées, 0 pour interdites
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask