from seq2seq_chat_model.models.attention import (
    AdditiveAttention,
    DotProductAttention,
)
import torch

# initialize example values
batch_size = 10
enc_seq_len = 5
dec_seq_len = 7
enc_dim = 256
dec_dim = 128
enc_seq = torch.ones((batch_size, enc_seq_len, enc_dim))
dec_seq = torch.ones((batch_size, dec_seq_len, dec_dim))
n_heads = 8


def test_additive_attention():
    """
    Tests, if additive attention returns context of right shape.
    """
    attention = AdditiveAttention(enc_dim, dec_dim, n_heads=n_heads)
    context = attention(enc_seq, enc_seq, dec_seq)
    assert context.shape == torch.Size([batch_size, dec_seq_len, enc_dim])


def test_dot_product_attention():
    """
    Tests, if (scaled) dot-product attention returns context of right shape.
    """
    attention = DotProductAttention(
        enc_dim, dec_dim, n_heads=n_heads, scaled=True
    )
    context = attention(enc_seq, enc_seq, dec_seq)
    assert context.shape == torch.Size([batch_size, dec_seq_len, enc_dim])

    attention = DotProductAttention(
        enc_dim, dec_dim, n_heads=n_heads, scaled=False
    )
    context = attention(enc_seq, enc_seq, dec_seq)
    assert context.shape == torch.Size([batch_size, dec_seq_len, enc_dim])


if __name__ == "__main__":
    test_additive_attention()
    test_dot_product_attention()
