import torch
from transformer import Embedding
from transformer import PositionalEncoding
from transformer import ScaledDotProductAttention
from transformer import MultiHeadAttention


def create_causal_mask(seq_len):
    """
    Crée un masque causal (triangulaire inférieur) pour l'auto-attention masquée
    Empêche de voir les tokens futurs
    Retourne: (seq_len, seq_len) - 1 pour positions autorisées, 0 pour interdites
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

def main():
    # Test Embedding
    print("Embedding ================================================\n")
    batch_size, vocab_size, seq_len, model_dim = 1, 4, 4, 2
    embedding = Embedding(vocab_size, model_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding(input_ids)
    print(input_ids)
    print(output)
    print("\n")

    # Test PositionalEncoding
    print("PositionalEncoding =======================================\n")
    positional_encoding = PositionalEncoding(model_dim)
    output = positional_encoding(output)
    print(output)
    print("\n")

    # Test ScaledDotProductAttention
    print("ScaledDotProductAttention ================================\n")
    batch_size, seq_len, key_dim, value_dim = 1, 4, model_dim, model_dim
    attention = ScaledDotProductAttention(key_dim, value_dim)
    queries = torch.randn(batch_size, seq_len, key_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    attention_output, weights = attention(queries, keys, values)
    print(queries)
    print(attention_output)
    print(weights)
    print("\n")

    print("ScaledDotProductAttention on positional encoding =========\n")
    attention_output, weights = attention(output, output, output)
    print(attention_output)
    print(weights)
    print("\n")

    print("ScaledDotProductAttention with mask ======================\n")
    mask = create_causal_mask(seq_len)
    attention = ScaledDotProductAttention(key_dim, value_dim)
    attention_output, weights = attention(output, output, output, mask)
    print(attention_output)
    print(weights)
    print("\n")

    # print("MultiHeadAttention =======================================\n") bug
    # number_of_heads = 4
    # mha = MultiHeadAttention(model_dim, number_of_heads)
    # mha_output = mha(output, output, output)
    # print(mha_output)
    # print("\n")
    
if __name__ == "__main__":
    main()