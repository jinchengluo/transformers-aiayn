import torch
from transformer import Embedding
from transformer import PositionalEncoding
from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention
from transformer import PositionwiseFeedForwardNetwork
from transformer import EncoderLayer, Encoder
from transformer import DecoderLayer, Decoder
from transformer import Transformer

def create_causal_mask(sequence_length):
    """Create causal (look-ahead) mask for decoder self-attention"""
    mask = torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)
    return mask == 0

def main():

    # Test parameters
    batch_size = 1
    src_seq_len = 4
    tgt_seq_len = 4
    d_model = 8
    src_vocab_size = 8
    tgt_vocab_size = 8
    h = 8
    d_ff = 2048
    N = 6 
    #torch.manual_seed(42)


    print(f"Test Parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Source sequence length: {src_seq_len}")
    print(f"  Target sequence length: {tgt_seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Source vocab size: {src_vocab_size}")
    print(f"  Target vocab size: {tgt_vocab_size}")
    print(f"  Number of heads: {h}")
    print(f"  Number of layers: {N}")
    print()

    # Test Embedding
    print("\nEmbedding ================================================\n")
    embedding = Embedding(src_vocab_size, d_model)

    input_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    input_embedded = embedding(input_ids)

    print(input_ids)
    print(input_embedded)
    print()

    # Test PositionalEncoding
    print("PositionalEncoding =======================================\n")
    positional_encoding = PositionalEncoding(d_model)

    input_pos_encoded = positional_encoding(input_embedded)

    print(input_pos_encoded)
    print()

    # Test ScaledDotProductAttention
    print("ScaledDotProductAttention ================================\n")
    attention = ScaledDotProductAttention(d_model, d_model)

    attention_output, weights = attention(input_pos_encoded, input_pos_encoded, input_pos_encoded)

    print(attention_output)
    print(weights)
    print()

    print("ScaledDotProductAttention with mask ======================\n")
    mask = create_causal_mask(src_seq_len)
    
    mask_attention_output, mask_weights = attention(input_pos_encoded, input_pos_encoded, input_pos_encoded, mask)

    print(mask)
    print(mask_attention_output)
    print(mask_weights)
    print()

    # Test MultiHeadAttention
    print("MultiHeadAttention =======================================\n")
    mha = MultiHeadAttention(d_model, h, True)

    mha_output = mha(input_pos_encoded, input_pos_encoded, input_pos_encoded)
    
    print(mha_output)
    print(mha.attention_weigths)
    print()

    print("MultiHeadAttention with mask ==============================\n")    
    mask_mha_output = mha(input_pos_encoded, input_pos_encoded, input_pos_encoded, mask)
    
    print(mask_mha_output)
    print(mha.attention_weigths)
    print()

    # Test PositionwiseFeedForwardNetwork
    print("PositionalwiseFeedForwardNetwork =========================\n")
    ffn = PositionwiseFeedForwardNetwork(d_model, d_ff)
    
    ffn_output = ffn(mha_output)
    
    print(ffn_output)
    print()

    # Test EncoderLayer
    print("EncoderLayer ============================================\n")
    encoder_layer = EncoderLayer(d_model, mha, ffn)
    
    encoder_layer_output = encoder_layer(input_pos_encoded)
    
    print(encoder_layer_output)
    print()

    # Test Encoder
    print("Encoder ==================================================\n")
    encoder = Encoder(d_model, N, h, d_ff)
    
    input_encoded = encoder(input_pos_encoded)
    
    print(input_encoded)
    print()
    
    # Test DecoderLayer
    print("DecoderLayer ============================================\n")
    embedding = Embedding(tgt_vocab_size, d_model)
    positional_encoding = PositionalEncoding(d_model)
    
    output_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    output_embedded = embedding(output_ids)
    output_pos_encoded = positional_encoding(output_embedded)
    
    masked_mha = MultiHeadAttention(d_model, h)
    decoder_layer = DecoderLayer(d_model, masked_mha, mha, ffn)
    
    decoder_layer_output = decoder_layer(input_encoded, output_pos_encoded, mask)
    
    print(decoder_layer_output)
    print()

    # Test Decoder
    print("Decoder ==================================================\n")
    decoder = Decoder(d_model, N, h, d_ff)

    output_decoded = decoder(input_encoded, output_pos_encoded, mask)
    
    print(output_decoded)
    print()

    # Test Transformer
    print("Transformer ==============================================\n")
    transformer = Transformer(d_model, d_ff, N, h, src_vocab_size, tgt_vocab_size)

    transformer_output = transformer(input_ids, output_ids, mask)
    
    print(transformer_output)
    print()
    
    
if __name__ == "__main__":
    main()