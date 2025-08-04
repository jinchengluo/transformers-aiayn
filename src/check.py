import torch
from transformer import create_causal_mask
from transformer import Embedding
from transformer import PositionalEncoding
from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention
from transformer import PositionwiseFeedForwardNetwork
from transformer import EncoderLayer, Encoder
from transformer import DecoderLayer, Decoder
from transformer import Transformer

def main():
    # Test Embedding
    print("\nEmbedding ================================================\n")
    batch_size, vocab_size, seq_len, model_dim = 1, 4, 4, 8
    embedding = Embedding(vocab_size, model_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_embedded = embedding(input_ids)
    print(input_ids)
    print(input_embedded)
    print("\n")

    # Test PositionalEncoding
    print("PositionalEncoding =======================================\n")
    positional_encoding = PositionalEncoding(model_dim)
    input_pos_encoded = positional_encoding(input_embedded)
    print(input_pos_encoded)
    print("\n")

    # Test ScaledDotProductAttention
    print("ScaledDotProductAttention ================================\n")
    attention = ScaledDotProductAttention(model_dim, model_dim)
    attention_output, weights = attention(input_pos_encoded, input_pos_encoded, input_pos_encoded)
    print(attention_output)
    print(weights)
    print("\n")

    print("ScaledDotProductAttention with mask ======================\n")
    mask = create_causal_mask(seq_len)
    mask_attention_output, mask_weights = attention(input_pos_encoded, input_pos_encoded, input_pos_encoded, mask)
    print(mask_attention_output)
    print(mask_weights)
    print("\n")

    # Test MultiHeadAttention
    print("MultiHeadAttention =======================================\n")
    number_of_heads = 4
    mha = MultiHeadAttention(model_dim, number_of_heads)
    mha_output = mha(input_pos_encoded, input_pos_encoded, input_pos_encoded)
    print(mha_output)
    print("\n")

    print("MultiHeadAttention with mask ==============================\n")
    mask = create_causal_mask(seq_len)
    mask_mha_output = mha(input_pos_encoded, input_pos_encoded, input_pos_encoded, mask)
    print(mask_mha_output)
    print("\n")

    # Test PositionwiseFeedForwardNetwork
    print("PositionalwiseFeedForwardNetwork =========================\n")
    ff_dim = 8
    ffn = PositionwiseFeedForwardNetwork(model_dim, ff_dim)
    ffn_output = ffn(mha_output)
    print(ffn_output)
    print("\n")

    # Test EncoderLayer
    print("EncoderLayer ============================================\n")
    encoder_layer = EncoderLayer(model_dim, mha, ffn)
    encoder_layer_output = encoder_layer(input_pos_encoded)
    print(encoder_layer_output)
    print("\n")

    # Test Encoder
    print("Encoder ==================================================\n")
    number_of_layers = 6
    encoder = Encoder(model_dim, number_of_layers, encoder_layer)
    input_encoded = encoder(input_pos_encoded)
    print(input_encoded)
    print("\n")
    
    # Test DecoderLayer
    print("DecoderLayer ============================================\n")
    embedding = Embedding(vocab_size, model_dim)
    positional_encoding = PositionalEncoding(model_dim)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output_embedded = embedding(output_ids)
    output_pos_encoded = positional_encoding(output_embedded)
    masked_mha = MultiHeadAttention(model_dim, number_of_heads)
    decoder_layer = DecoderLayer(model_dim, masked_mha, mha, ffn)
    decoder_layer_output = decoder_layer(input_encoded, output_pos_encoded, mask)
    print(decoder_layer_output)
    print("\n")

    # Test Decoder
    print("Decoder ==================================================\n")
    decoder = Decoder(model_dim, number_of_layers, decoder_layer)
    output_decoded = decoder(input_encoded, output_pos_encoded, mask)
    print(output_decoded)
    print("\n")

    # Test Transformer
    print("Transformer ==============================================\n")
    transformer = Transformer(model_dim, ff_dim, number_of_layers, number_of_heads, vocab_size, vocab_size)
    transformer_output = transformer(input_ids, output_ids, mask)
    print(transformer_output)
    print("\n")
    
if __name__ == "__main__":
    main()