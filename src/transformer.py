import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, model_dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.model_dimension = model_dimension
        self.embedding = nn.Embedding(vocabulary_size, model_dimension)
    
    def forward(self, src_token_ids):
        return self.embedding(src_token_ids) * math.sqrt(self.model_dimension)
    

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

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)       

    def forward(self, src_embedded):
        return src_embedded + self.positional_encoding[:, :src_embedded.shape[1]]


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
    def __init__(self, model_dimension, number_of_heads, save_weigths=False):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.key_dimension = model_dimension // number_of_heads
        self.value_dimension = model_dimension // number_of_heads

        self.linear_queries = nn.Linear(model_dimension, model_dimension) # W_Q
        self.linear_keys = nn.Linear(model_dimension, model_dimension) # W_K
        self.linear_values = nn.Linear(model_dimension, model_dimension) # W_V

        self.linear_output = nn.Linear(model_dimension, model_dimension) # w_0
        
        self.attention_weigths = None
        self.save_weigths = save_weigths

    def forward(self, queries, keys, values, mask=None):         
        projected_queries = self.linear_queries(queries) # Q * W_Q
        projected_queries = projected_queries.view(projected_queries.shape[0], projected_queries.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_keys = self.linear_keys(keys) # K * W_K
        projected_keys = projected_keys.view(projected_keys.shape[0], projected_keys.shape[1], self.number_of_heads, self.key_dimension).transpose(1, 2)

        projected_values = self.linear_values(values) # V * W_V
        projected_values = projected_values.view(projected_values.shape[0], projected_values.shape[1], self.number_of_heads, self.value_dimension).transpose(1, 2)
    
        scaled_dot_product = ScaledDotProductAttention(self.key_dimension, self.value_dimension)

        attention, attention_weigths = scaled_dot_product(projected_queries, projected_keys, projected_values, mask)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(queries.shape[0], -1, self.number_of_heads * self.key_dimension)

        if self.save_weigths :
            self.attention_weigths = attention_weigths.detach()

        return self.linear_output(attention)


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimension, inner_layer_dimension):
        super().__init__()
        self.modul_dimension = model_dimension
        self.inner_layer_dimension = inner_layer_dimension
        
        self.linear1 = nn.Linear(model_dimension, inner_layer_dimension) # W1 and b1
        self.linear2 = nn.Linear(inner_layer_dimension, model_dimension) # W2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, multihead_attention, feedforward_network):
        super().__init__()
        self.model_dimension = model_dimension
        
        self.multihead_attention = multihead_attention
        self.feedforward_network = feedforward_network

        self.layernorm1 = nn.LayerNorm(model_dimension)
        self.layernorm2 = nn.LayerNorm(model_dimension)

    def forward(self, src_embedded, src_mask=None):
        attention_ouput = self.multihead_attention(src_embedded, src_embedded, src_embedded, src_mask)
        sublayer_output1 = self.layernorm1(src_embedded + attention_ouput)

        ffnetwork_output = self.feedforward_network(sublayer_output1)
        sublayer_output2 = self.layernorm2(sublayer_output1 + ffnetwork_output)
        
        return sublayer_output2
    

class Encoder(nn.Module):
    def __init__(self, model_dimension, number_of_layers, number_of_heads, inner_layer_dimension):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                model_dimension,
                MultiHeadAttention(model_dimension, number_of_heads),
                PositionwiseFeedForwardNetwork(model_dimension, inner_layer_dimension)
            )
            for _ in range(number_of_layers)
        ])

    def forward(self, src_embedded, src_mask=None):
        src_encoder_output = src_embedded

        for layer in self.layers:
            src_encoder_output = layer(src_encoder_output, src_mask)
        
        return src_encoder_output
    

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

    def forward(self, src_encoder_output, trg_embedded, src_mask=None, trg_mask=None):
        masked_attention_output = self.masked_multihead_attention(queries=trg_embedded, keys=trg_embedded, values=trg_embedded, mask=trg_mask)
        sublayer_output1 = self.layernorm1(trg_embedded + masked_attention_output)
        
        attention_ouput = self.multihead_attention(queries=sublayer_output1, keys=src_encoder_output, values=src_encoder_output, mask=src_mask)
        sublayer_output2 = self.layernorm2(sublayer_output1 + attention_ouput)
        
        ffnetwork_ouput = self.feedforward_network(sublayer_output2)
        sublayer_output3 = self.layernorm3(sublayer_output2 + ffnetwork_ouput)
        
        return sublayer_output3
    

class Decoder(nn.Module):
    def __init__(self, model_dimension, number_of_layers, number_of_heads, inner_layer_dimension):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers

        self.layers = nn.ModuleList([
            DecoderLayer(
                model_dimension,
                MultiHeadAttention(model_dimension, number_of_heads),  # masked self-attention
                MultiHeadAttention(model_dimension, number_of_heads),  # encoder-decoder attention
                PositionwiseFeedForwardNetwork(model_dimension, inner_layer_dimension)
            )
            for _ in range(number_of_layers)
        ])
    
    def forward(self, src_encoder_output, trg_embedded, src_mask=None, trg_mask=None):
        trg_decoder_output = trg_embedded

        for layer in self.layers:
            trg_decoder_output = layer(src_encoder_output, trg_decoder_output, src_mask=src_mask, trg_mask=trg_mask)
        
        return trg_decoder_output


class Transformer(nn.Module):
    def __init__(self, model_dimension, inner_layer_dimension, number_of_layers, number_of_heads, src_vocabulary_size, trg_vocabulary_size):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.src_vocabulary_size = src_vocabulary_size
        self.trg_vocabulary_size = trg_vocabulary_size

        self.input_embedding = Embedding(src_vocabulary_size, model_dimension)
        self.output_embedding = Embedding(trg_vocabulary_size, model_dimension)

        self.input_pos_encoding = PositionalEncoding(model_dimension)
        self.output_pos_encoding = PositionalEncoding(model_dimension)

        self.encoder = Encoder(model_dimension, number_of_layers, number_of_heads, inner_layer_dimension)
        self.decoder = Decoder(model_dimension, number_of_layers, number_of_heads, inner_layer_dimension)

        self.linear_projection = nn.Linear(model_dimension, trg_vocabulary_size)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, src_token_ids, src_mask=None):
        input_embedded = self.input_embedding(src_token_ids)
        input_pos_encoded = self.input_pos_encoding(input_embedded)
        input_encoded = self.encoder(input_pos_encoded, src_mask)
        return input_encoded
    
    def decode(self, input_encoded, trg_token_ids, src_mask, trg_mask=None):
        output_embedded = self.output_embedding(trg_token_ids)
        output_pos_encoded = self.output_pos_encoding(output_embedded)
        output_decoded = self.decoder(input_encoded, output_pos_encoded, src_mask, trg_mask)
        return output_decoded

    def forward(self, src_token_ids, trg_token_ids, src_mask=None, trg_mask=None):
        input_embedded = self.input_embedding(src_token_ids)
        input_pos_encoded = self.input_pos_encoding(input_embedded)
        input_encoded = self.encoder(input_pos_encoded, src_mask)

        output_embedded = self.output_embedding(trg_token_ids)
        output_pos_encoded = self.output_pos_encoding(output_embedded)
        output_decoded = self.decoder(input_encoded, output_pos_encoded, src_mask, trg_mask)
        
        output_decoded = self.linear_projection(output_decoded)
        # output_decoded = output_decoded.softmax(dim=-1)

        return output_decoded
    

### Test functions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())

    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')
        

if __name__ == "__main__":
    torch.manual_seed(42)
    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=512,
        src_vocabulary_size=src_vocab_size,
        trg_vocabulary_size=trg_vocab_size,
        number_of_heads=8,
        number_of_layers=6,
        inner_layer_dimension=2048
    )

    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the baseline transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)

    print(out)