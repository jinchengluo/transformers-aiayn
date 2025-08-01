import math
import torch
import torch.nn as nn
from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention
from transformer import PositionwiseFeedForwardNetwork
from transformer import Embedding
from transformer import PositionalEncoding
from transformer import EncoderLayer
from transformer import Encoder
from transformer import DecoderLayer
from transformer import Decoder
from transformer import Transformer

def test_scaled_dot_product_attention():
    print("Testing ScaledDotProductAttention...")
    
    batch_size, seq_len, key_dim, value_dim = 2, 4, 8, 8
    attention = ScaledDotProductAttention(key_dim, value_dim)
    
    queries = torch.randn(batch_size, seq_len, key_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    output, weights = attention(queries, keys, values)
    
    assert output.shape == (batch_size, seq_len, value_dim), f"Expected shape {(batch_size, seq_len, value_dim)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Expected weights shape {(batch_size, seq_len, seq_len)}, got {weights.shape}"
    
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
    
    assert output.shape == x.shape, f"Expected same shape as input {x.shape}, got {output.shape}"
    
    intermediate = ffn.relu(ffn.linear1(x))
    assert (intermediate >= 0).all(), "ReLU should produce non-negative values"
    
    print("PositionwiseFeedForwardNetwork tests passed")


def test_embedding():
    print("Testing Embedding...")
    
    vocab_size, model_dim = 1000, 512
    embedding = Embedding(vocab_size, model_dim)
    
    batch_size, seq_len = 2, 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding(input_ids)
    
    assert output.shape == (batch_size, seq_len, model_dim), f"Expected shape {(batch_size, seq_len, model_dim)}, got {output.shape}"
    
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
    
    assert output.shape == x_embedded.shape, f"Expected same shape as input {x_embedded.shape}, got {output.shape}"
    
    pos_encoding_slice = pos_enc.positional_encoding[:seq_len]
    expected_output = x_embedded + pos_encoding_slice
    assert torch.allclose(output, expected_output), "Positional encoding should be added to input"
    
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
    
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output = decoder_layer(encoder_output, decoder_input, mask)
    
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
    
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output = decoder(encoder_output, decoder_input, mask)
    
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
    
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, -1] = 0
    
    output, weights = attention(queries, keys, values, mask)
    
    assert torch.all(weights[:, :, -1] < 1e-6), "Masked positions should have near-zero attention weights"
    
    print("Attention mask tests passed")


def test_transformer():
    """Test complet de la classe Transformer"""
    print("Testing Transformer...")
    
    # Paramètres du modèle
    model_dim = 512
    inner_dim = 2048
    num_layers = 6
    num_heads = 8
    input_vocab_size = 10000
    output_vocab_size = 10000
    
    # Créer le modèle
    transformer = Transformer(
        model_dimension=model_dim,
        inner_layer_dimension=inner_dim,
        number_of_layers=num_layers,
        number_of_heads=num_heads,
        input_vocabulary_size=input_vocab_size,
        output_vocabulary_size=output_vocab_size    
    )
    
    # Données de test
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Créer des séquences d'entrée et de sortie
    input_ids = torch.randint(0, input_vocab_size, (batch_size, src_seq_len))
    output_ids = torch.randint(0, output_vocab_size, (batch_size, tgt_seq_len))
    
    # Créer un masque causal pour le décodeur
    causal_mask = transformer.generate_causal_mask(tgt_seq_len)
    
    # Test du forward pass
    logits = transformer(input_ids, output_ids, causal_mask)
    
    # Vérifications des dimensions
    expected_shape = (batch_size, tgt_seq_len, output_vocab_size)
    assert logits.shape == expected_shape, f"Expected output shape {expected_shape}, got {logits.shape}"
    
    # Vérifier que les logits sont des nombres réels (pas NaN ou inf)
    assert torch.isfinite(logits).all(), "Output logits should be finite"
    
    # Test avec différentes longueurs de séquence
    src_seq_len2 = 15
    tgt_seq_len2 = 12
    input_ids2 = torch.randint(0, input_vocab_size, (batch_size, src_seq_len2))
    output_ids2 = torch.randint(0, output_vocab_size, (batch_size, tgt_seq_len2))
    causal_mask2 = transformer.generate_causal_mask(tgt_seq_len2)
    
    logits2 = transformer(input_ids2, output_ids2, causal_mask2)
    expected_shape2 = (batch_size, tgt_seq_len2, output_vocab_size)
    assert logits2.shape == expected_shape2, f"Expected output shape {expected_shape2}, got {logits2.shape}"
    
    print("Transformer basic functionality tests passed")


def test_transformer_components_integration():
    """Test que tous les composants du Transformer s'intègrent correctement"""
    print("Testing Transformer components integration...")
    
    model_dim = 256
    inner_dim = 1024
    num_layers = 2
    num_heads = 4
    input_vocab_size = 1000
    output_vocab_size = 1000
    
    transformer = Transformer(
        model_dimension=model_dim,
        inner_layer_dimension=inner_dim,
        number_of_layers=num_layers,
        number_of_heads=num_heads,
        input_vocabulary_size=input_vocab_size,
        output_vocabulary_size=output_vocab_size
    )
    
    batch_size = 1
    seq_len = 5
    
    input_ids = torch.randint(0, input_vocab_size, (batch_size, seq_len))
    output_ids = torch.randint(0, output_vocab_size, (batch_size, seq_len))
    
    # Test sans masque
    logits_no_mask = transformer(input_ids, output_ids)
    assert logits_no_mask.shape == (batch_size, seq_len, output_vocab_size)
    
    # Test avec masque
    mask = transformer.generate_causal_mask(seq_len)
    logits_with_mask = transformer(input_ids, output_ids, mask)
    assert logits_with_mask.shape == (batch_size, seq_len, output_vocab_size)
    
    # Les résultats avec et sans masque devraient être différents
    assert not torch.allclose(logits_no_mask, logits_with_mask, atol=1e-6), "Mask should affect the output"
    
    print("Transformer integration tests passed")


def test_transformer_causal_mask():
    """Test la génération et l'utilisation des masques causaux"""
    print("Testing Transformer causal mask...")
    
    transformer = Transformer(512, 2048, 2, 8, 1000, 1000)
    
    # Test de génération de masque
    seq_len = 4
    mask = transformer.generate_causal_mask(seq_len)
    
    expected_mask = torch.tensor([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ])
    
    assert mask.shape == (1, 1, seq_len, seq_len), f"Expected mask shape (1, 1, {seq_len}, {seq_len}), got {mask.shape}"
    assert torch.allclose(mask.squeeze(), expected_mask), "Causal mask should be lower triangular"
    
    print("Transformer causal mask tests passed")


def test_transformer_gradient_flow():
    """Test que les gradients se propagent correctement"""
    print("Testing Transformer gradient flow...")
    
    model_dim = 128
    transformer = Transformer(
        model_dimension=model_dim,
        inner_layer_dimension=512,
        number_of_layers=2,
        number_of_heads=4,
        input_vocabulary_size=100,
        output_vocabulary_size=100
    )
    
    batch_size = 2
    seq_len = 5
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    output_ids = torch.randint(0, 100, (batch_size, seq_len))
    target = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    logits = transformer(input_ids, output_ids)
    
    # Calculer une loss simple
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 100), target.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Vérifier que les gradients existent
    has_gradients = False
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            has_gradients = True
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or inf"
    
    assert has_gradients, "At least some parameters should have gradients"
    
    print("Transformer gradient flow tests passed")


def test_transformer_parameter_count():
    """Test le nombre de paramètres du modèle"""
    print("Testing Transformer parameter count...")
    
    model_dim = 512
    transformer = Transformer(
        model_dimension=model_dim,
        inner_layer_dimension=2048,
        number_of_layers=6,
        number_of_heads=8,
        input_vocabulary_size=10000,
        output_vocabulary_size=10000
    )
    
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    # Vérifier que le modèle a des paramètres
    assert total_params > 0, "Model should have parameters"
    assert trainable_params > 0, "Model should have trainable parameters"
    assert total_params == trainable_params, "All parameters should be trainable by default"
    
    print(f"Transformer has {total_params:,} parameters ({trainable_params:,} trainable)")


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

        print("\n=== RUNNING TRANSFORMER MODEL TESTS ===\n")
        test_transformer()
        test_transformer_components_integration()
        test_transformer_causal_mask()
        test_transformer_gradient_flow()
        test_transformer_parameter_count()
        
        print("\nALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()