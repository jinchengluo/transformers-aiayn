import math
import torch
from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention
from transformer import PositionwiseFeedForwardNetwork
from transformer import Embedding
from transformer import PositionalEncoding
from transformer import EncoderLayer
from transformer import Encoder
from transformer import DecoderLayer
from transformer import Decoder

def test_scaled_dot_product_attention():
    print("Testing ScaledDotProductAttention...")
    
    batch_size, seq_len, key_dim, value_dim = 2, 4, 8, 8
    attention = ScaledDotProductAttention(key_dim, value_dim)
    
    queries = torch.randn(batch_size, seq_len, key_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    output, weights = attention(queries, keys, values)
    
    # Vérifications
    assert output.shape == (batch_size, seq_len, value_dim), f"Expected shape {(batch_size, seq_len, value_dim)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Expected weights shape {(batch_size, seq_len, seq_len)}, got {weights.shape}"
    
    # Les poids doivent sommer à 1 pour chaque position
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6), "Attention weights should sum to 1"
    
    print("ScaledDotProductAttention tests passed")


def test_multihead_attention():
    print("Testing MultiHeadAttention...")
    
    batch_size, seq_len, model_dim, num_heads = 2, 4, 512, 8
    mha = MultiHeadAttention(model_dim, num_heads)
    
    queries = torch.randn(batch_size, seq_len, model_dim)
    keys = torch.randn(batch_size, seq_len, model_dim)
    values = torch.randn(batch_size, seq_len, model_dim)
    
    output = mha(queries, keys, values)
    
    # Vérifications
    assert output.shape == (batch_size, seq_len, model_dim), f"Expected shape {(batch_size, seq_len, model_dim)}, got {output.shape}"
    assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"
    assert mha.key_dimension == model_dim // num_heads, "Key dimension should be model_dim // num_heads"
    
    print("MultiHeadAttention tests passed")


def test_feedforward_network():
    print("Testing PositionwiseFeedForwardNetwork...")
    
    batch_size, seq_len, model_dim, inner_dim = 2, 4, 512, 2048
    ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
    
    x = torch.randn(batch_size, seq_len, model_dim)
    output = ffn(x)
    
    # Vérifications
    assert output.shape == x.shape, f"Expected same shape as input {x.shape}, got {output.shape}"
    
    # Tester que ReLU fonctionne (pas de valeurs négatives dans la couche intermédiaire)
    intermediate = ffn.relu(ffn.linear1(x))
    assert (intermediate >= 0).all(), "ReLU should produce non-negative values"
    
    print("PositionwiseFeedForwardNetwork tests passed")


def test_embedding():
    print("Testing Embedding...")
    
    vocab_size, model_dim = 1000, 512
    embedding = Embedding(vocab_size, model_dim)
    
    batch_size, seq_len = 2, 10
    # Créer des indices valides (< vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding(input_ids)
    
    # Vérifications
    assert output.shape == (batch_size, seq_len, model_dim), f"Expected shape {(batch_size, seq_len, model_dim)}, got {output.shape}"
    
    # Vérifier le scaling par sqrt(model_dim)
    base_embedding = embedding.embedding(input_ids)
    expected_output = base_embedding * math.sqrt(model_dim)
    assert torch.allclose(output, expected_output), "Embedding should be scaled by sqrt(model_dim)"
    
    print("Embedding tests passed")


def test_positional_encoding():
    print("Testing PositionalEncoding...")
    
    model_dim, max_seq_len = 512, 5000
    pos_enc = PositionalEncoding(model_dim, max_seq_len)
    
    batch_size, seq_len = 2, 10
    x_embedded = torch.randn(batch_size, seq_len, model_dim)
    
    output = pos_enc(x_embedded)
    
    # Vérifications
    assert output.shape == x_embedded.shape, f"Expected same shape as input {x_embedded.shape}, got {output.shape}"
    
    # Vérifier que l'encodage positionnel est ajouté
    pos_encoding_slice = pos_enc.positional_encoding[:seq_len]
    expected_output = x_embedded + pos_encoding_slice
    assert torch.allclose(output, expected_output), "Positional encoding should be added to input"
    
    # Vérifier les propriétés de l'encodage positionnel
    pe = pos_enc.positional_encoding
    assert pe.shape == (max_seq_len, model_dim), "Positional encoding should have correct shape"
    
    print("PositionalEncoding tests passed")


def test_encoder_layer():
    print("Testing EncoderLayer...")
    
    model_dim, num_heads, inner_dim = 512, 8, 2048
    mha = MultiHeadAttention(model_dim, num_heads)
    ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
    encoder_layer = EncoderLayer(model_dim, mha, ffn)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, model_dim)
    
    output = encoder_layer(x)
    
    # Vérifications
    assert output.shape == x.shape, f"Expected same shape as input {x.shape}, got {output.shape}"
    
    print("EncoderLayer tests passed")


def test_encoder():
    print("Testing Encoder...")
    
    model_dim, num_heads, inner_dim, num_layers = 512, 8, 2048, 6
    mha = MultiHeadAttention(model_dim, num_heads)
    ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
    encoder_layer = EncoderLayer(model_dim, mha, ffn)
    encoder = Encoder(model_dim, num_layers, encoder_layer)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, model_dim)
    
    output = encoder(x)
    
    # Vérifications
    assert output.shape == x.shape, f"Expected same shape as input {x.shape}, got {output.shape}"
    assert len(encoder.layers) == num_layers, f"Expected {num_layers} layers, got {len(encoder.layers)}"
    
    print("Encoder tests passed")


def test_decoder_layer():
    print("Testing DecoderLayer...")
    
    model_dim, num_heads, inner_dim = 512, 8, 2048
    masked_mha = MultiHeadAttention(model_dim, num_heads)
    mha = MultiHeadAttention(model_dim, num_heads)
    ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
    decoder_layer = DecoderLayer(model_dim, masked_mha, mha, ffn)
    
    batch_size, seq_len = 2, 10
    encoder_output = torch.randn(batch_size, seq_len, model_dim)
    decoder_input = torch.randn(batch_size, seq_len, model_dim)
    
    # Créer un masque causal (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output = decoder_layer(encoder_output, decoder_input, mask)
    
    # Vérifications
    assert output.shape == decoder_input.shape, f"Expected same shape as decoder input {decoder_input.shape}, got {output.shape}"
    
    print("DecoderLayer tests passed")


def test_decoder():
    print("Testing Decoder...")
    
    model_dim, num_heads, inner_dim, num_layers = 512, 8, 2048, 6
    masked_mha = MultiHeadAttention(model_dim, num_heads)
    mha = MultiHeadAttention(model_dim, num_heads)
    ffn = PositionwiseFeedForwardNetwork(model_dim, inner_dim)
    decoder_layer = DecoderLayer(model_dim, masked_mha, mha, ffn)
    decoder = Decoder(model_dim, num_layers, decoder_layer)
    
    batch_size, seq_len = 2, 10
    encoder_output = torch.randn(batch_size, seq_len, model_dim)
    decoder_input = torch.randn(batch_size, seq_len, model_dim)
    
    # Créer un masque causal
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output = decoder(encoder_output, decoder_input, mask)
    
    # Vérifications
    assert output.shape == decoder_input.shape, f"Expected same shape as decoder input {decoder_input.shape}, got {output.shape}"
    assert len(decoder.layers) == num_layers, f"Expected {num_layers} layers, got {len(decoder.layers)}"
    
    print("Decoder tests passed")


def test_attention_mask():
    print("Testing attention masks...")
    
    batch_size, seq_len, key_dim = 2, 4, 8
    attention = ScaledDotProductAttention(key_dim, key_dim)
    
    queries = torch.randn(batch_size, seq_len, key_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, key_dim)
    
    # Masque qui bloque la dernière position
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, -1] = 0  # Bloquer la dernière colonne
    
    output, weights = attention(queries, keys, values, mask)
    
    # Vérifier que les poids pour la dernière position sont très petits (proche de 0)
    assert torch.all(weights[:, :, -1] < 1e-6), "Masked positions should have near-zero attention weights"
    
    print("Attention mask tests passed")


def run_all_tests():
    print("=== RUNNING TRANSFORMER COMPONENT TESTS ===\n")
    
    try:
        test_scaled_dot_product_attention()
        test_multihead_attention()
        test_feedforward_network()
        test_embedding()
        test_positional_encoding()
        test_encoder_layer()
        test_encoder()
        test_decoder_layer()
        test_decoder()
        test_attention_mask()
        
        print("\nALL TESTS PASSED!")
        print("Your Transformer components are working correctly!")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()