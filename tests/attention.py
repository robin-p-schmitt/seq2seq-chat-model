from seq2seq_chat_model.models.attention import (
    AdditiveAttention,
    DotProductAttention,
    ScaledDotProductAttention,
    CosineAttention,
    GeneralAttention,
)
import torch
import unittest


class TestAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.enc_seq_len = 5
        self.dec_seq_len = 7
        self.enc_dim = 256
        self.dec_dim = 128
        self.enc_seq = torch.ones(
            (self.batch_size, self.enc_seq_len, self.enc_dim)
        )
        self.dec_seq = torch.ones(
            (self.batch_size, self.dec_seq_len, self.dec_dim)
        )
        self.n_heads = 8

        self.modules = [
            AdditiveAttention,
            DotProductAttention,
            ScaledDotProductAttention,
            CosineAttention,
            GeneralAttention,
        ]

    def assert_context_shape(self, attention_mod):
        context = attention_mod(self.enc_seq, self.enc_seq, self.dec_seq)
        assert context.shape == torch.Size(
            [self.batch_size, self.dec_seq_len, self.enc_dim]
        )

    def test_attention_modules(self):
        for module in self.modules:
            with self.subTest(module=module):
                attention = module(
                    self.enc_dim, self.dec_dim, n_heads=self.n_heads
                )
                context = attention(self.enc_seq, self.enc_seq, self.dec_seq)
                self.assertSequenceEqual(
                    context.shape,
                    torch.Size(
                        [self.batch_size, self.dec_seq_len, self.enc_dim]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
