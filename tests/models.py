from seq2seq_chat_model.models.attention import (
    AdditiveAttention,
    DotProductAttention,
    ScaledDotProductAttention,
    CosineAttention,
    GeneralAttention,
)

from seq2seq_chat_model.models.encoders import LSTMEncoder

from seq2seq_chat_model.models.decoders import LSTMAttentionDecoder
import torch
import unittest


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.vocab_size = 512

        self.enc_seq_len = 5
        self.enc_dim = 256
        self.num_enc_layers = 5
        self.enc_inp = torch.ones(self.batch_size, self.enc_seq_len).long()
        self.enc_out = torch.ones(
            (self.batch_size, self.enc_seq_len, self.enc_dim)
        )

        self.dec_dim = 128
        self.dec_seq_len = 7
        self.num_dec_layers = 5
        self.dec_inp = torch.ones(self.batch_size, self.dec_seq_len).long()
        self.dec_seq = torch.ones(
            (self.batch_size, self.dec_seq_len, self.dec_dim)
        )
        self.dec_out = torch.ones(
            (self.batch_size, self.dec_seq_len, self.vocab_size)
        )

        self.n_attention_heads = 8

        self.attentions = [
            AdditiveAttention,
            DotProductAttention,
            ScaledDotProductAttention,
            CosineAttention,
            GeneralAttention,
        ]

        self.encoders = [LSTMEncoder]

        self.decoders = [LSTMAttentionDecoder]

    def test_attention_modules(self):
        for module in self.attentions:
            with self.subTest(module=module):
                attention = module(
                    self.enc_dim, self.dec_dim, n_heads=self.n_attention_heads
                )
                context = attention(self.enc_out, self.enc_out, self.dec_seq)
                self.assertSequenceEqual(
                    context.shape,
                    torch.Size(
                        [self.batch_size, self.dec_seq_len, self.enc_dim]
                    ),
                )

    def test_encoders(self):
        for module in self.encoders:
            with self.subTest(encoder=module):
                encoder = module(
                    self.vocab_size, self.enc_dim, self.num_enc_layers
                )
                hidden = cell = encoder.init_hidden(self.batch_size)

                out, hidden, cell = encoder(self.enc_inp, hidden, cell)

                self.assertSequenceEqual(
                    out.shape,
                    torch.Size(
                        [self.batch_size, self.enc_seq_len, self.enc_dim * 2]
                    ),
                )

                self.assertSequenceEqual(
                    hidden.shape,
                    torch.Size(
                        [
                            self.num_enc_layers * 2,
                            self.batch_size,
                            self.enc_dim,
                        ]
                    ),
                )

                self.assertSequenceEqual(
                    cell.shape,
                    torch.Size(
                        [
                            self.num_enc_layers * 2,
                            self.batch_size,
                            self.enc_dim,
                        ]
                    ),
                )

    def test_decoders(self):
        for module in self.decoders:
            with self.subTest(decoder=module):
                attention = self.attentions[0](
                    self.enc_dim, self.dec_dim, n_heads=self.n_attention_heads
                )
                decoder = module(
                    self.dec_dim,
                    self.vocab_size,
                    self.enc_dim,
                    self.num_dec_layers,
                    attention,
                )
                hidden = cell = decoder.init_hidden((self.batch_size))

                out, hidden, cell = decoder(
                    self.dec_inp, hidden, cell, self.enc_out
                )

                self.assertSequenceEqual(
                    out.shape,
                    torch.Size(
                        [self.batch_size, self.dec_seq_len, self.vocab_size]
                    ),
                )
                self.assertSequenceEqual(
                    hidden.shape,
                    torch.Size(
                        [self.num_dec_layers, self.batch_size, self.dec_dim]
                    ),
                )
                self.assertSequenceEqual(
                    cell.shape,
                    torch.Size(
                        [self.num_dec_layers, self.batch_size, self.dec_dim]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
