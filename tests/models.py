"""This module tests all the functionality of the ``model`` subpackage.
"""

from seq2seq_chat_model.models.attention import (
    AdditiveAttention,
    DotProductAttention,
    ScaledDotProductAttention,
    CosineAttention,
    GeneralAttention,
)

from seq2seq_chat_model.models.encoders import LSTMEncoder, TransformerEncoder

from seq2seq_chat_model.models.decoders import (
    LSTMAttentionDecoder,
    TransformerDecoder,
)
import torch
import unittest


class TestAttention(unittest.TestCase):
    """Test different attention mechanisms."""

    def setUp(self) -> None:
        self.batch_size = 10

        self.query_size = 128
        self.query_len = 5
        self.query_seq = torch.ones(
            self.batch_size, self.query_len, self.query_size
        )

        self.key_val_size = 256
        self.key_val_len = 7
        self.key_val_seq = torch.ones(
            self.batch_size, self.key_val_len, self.key_val_size
        )

        self.d_k = 12
        self.d_v = 12

        self.n_heads = 8
        self.n_heads_error = 7

        # all modules to be tested
        self.attention_mod = [
            AdditiveAttention,
            DotProductAttention,
            ScaledDotProductAttention,
            CosineAttention,
            GeneralAttention,
        ]

    def test_attention(self):
        """Test attention between encoder and decoder sequence."""
        for module in self.attention_mod:
            with self.subTest(module=module):
                attention = module(
                    d_key_val=self.key_val_size,
                    len_key_val=self.key_val_len,
                    d_query=self.query_size,
                    len_query=self.query_len,
                    n_heads=self.n_heads,
                )
                context = attention(
                    self.key_val_seq, self.key_val_seq, self.query_seq
                )
                self.assertSequenceEqual(
                    context.shape,
                    torch.Size(
                        [self.batch_size, self.query_len, self.key_val_size]
                    ),
                )

    def test_self_attention(self):
        """Test self-attention mechanism."""
        for module in self.attention_mod:
            with self.subTest(module=module):
                attention = module(
                    d_key_val=self.query_size,
                    len_key_val=self.query_len,
                    d_query=self.query_size,
                    len_query=self.query_len,
                    n_heads=self.n_heads,
                )
                context = attention(
                    self.query_seq, self.query_seq, self.query_seq
                )
                self.assertSequenceEqual(
                    context.shape,
                    torch.Size(
                        [self.batch_size, self.query_len, self.query_size]
                    ),
                )

    def test_masked_attention(self):
        """Test masked attention mechanisms."""
        for module in self.attention_mod:
            with self.subTest(module=module):
                attention = module(
                    d_key_val=self.query_size,
                    len_key_val=self.query_len,
                    d_query=self.query_size,
                    len_query=self.query_len,
                    n_heads=self.n_heads,
                    masked=True,
                )
                attention(self.query_seq, self.query_seq, self.query_seq)

                scores = attention.get_attention_scores()
                for i in range(scores.shape[0]):
                    for j in range(scores.shape[1]):
                        zero_score = torch.sum(
                            scores[i, j, j + 1 :], dim=-1
                        ).squeeze()
                        self.assertAlmostEqual(zero_score.item(), 0)

    def test_n_heads_error(self):
        """Test error in case of not matching n_heads and d_k/d_v"""
        for module in self.attention_mod:
            with self.subTest(module=module):
                with self.assertRaises(ValueError):
                    module(
                        d_key_val=self.query_size,
                        len_key_val=self.query_len,
                        d_query=self.query_size,
                        len_query=self.query_len,
                        n_heads=self.n_heads_error,
                    )

    def test_mask_error(self):
        """Test error in case of masked attention for different length sequences."""
        for module in self.attention_mod:
            with self.subTest(module=module):
                with self.assertRaises(ValueError):
                    module(
                        d_key_val=self.key_val_size,
                        len_key_val=self.key_val_len,
                        d_query=self.query_size,
                        len_query=self.query_len,
                        n_heads=self.n_heads,
                        masked=True,
                    )


class TestEncoders(unittest.TestCase):
    """Test all encoder modules."""

    def setUp(self) -> None:
        self.batch_size = 10
        self.vocab_size = 1048
        self.seq_len = 5

        self.input = torch.ones(self.batch_size, self.seq_len).long()

        self.hidden_size = 128
        self.num_layers = 4

    def test_lstm_encoder(self):
        encoder = LSTMEncoder(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        hidden = cell = encoder.init_hidden(self.batch_size)

        out, hidden, cell = encoder(self.input, hidden, cell)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.seq_len, self.hidden_size * 2]),
        )

    def test_transformer_encoder(self):
        encoder = TransformerEncoder(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
        )

        out = encoder(self.input)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.seq_len, self.hidden_size]),
        )

    def test_pretrained_emb(self):
        """Test if pretrained embeddings can be used."""
        embedding = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        LSTMEncoder(
            self.vocab_size, self.hidden_size, self.num_layers, embedding
        )


class TestDecoders(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.vocab_size = 1048
        self.dec_seq_len = 5
        self.enc_seq_len = 7

        self.dec_hidden_size = 128
        self.enc_hidden_size = 256
        self.num_layers = 4

        self.dec_input = torch.ones(self.batch_size, self.dec_seq_len).long()
        self.enc_out = torch.ones(
            self.batch_size, self.enc_seq_len, self.enc_hidden_size
        )

    def test_lstm_adaper(self):
        """Test if decoder works with different sized encoder states."""
        decoder = LSTMAttentionDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.enc_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
        )
        hidden = cell = decoder.init_hidden(self.batch_size)

        out, hidden, cell = decoder(self.dec_input, hidden, cell, self.enc_out)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )

    def test_transformer_adapter(self):
        """Test if decoder works with different sized encoder states."""
        decoder = TransformerDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.enc_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
        )

        out = decoder(self.dec_input, self.enc_out)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )

    def test_pretrained_emb_lstm(self):
        """Test if pretrained embeddings can be used."""
        embedding = torch.nn.Embedding(self.vocab_size, self.dec_hidden_size)
        decoder = LSTMAttentionDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.enc_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=AdditiveAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
            pretrained_emb=embedding,
        )

        hidden = cell = decoder.init_hidden(self.batch_size)

        out, hidden, cell = decoder(self.dec_input, hidden, cell, self.enc_out)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )

    def test_pretrained_emb_transformer(self):
        """Test if pretrained embeddings can be used."""
        embedding = torch.nn.Embedding(self.vocab_size, self.dec_hidden_size)
        decoder = TransformerDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.enc_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
            pretrained_emb=embedding,
        )

        out = decoder(self.dec_input, self.enc_out)

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )

    def test_lstm_no_adapter(self):
        """Test if decoder works with same sized encoder states."""
        decoder = LSTMAttentionDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.dec_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
        )
        hidden = cell = decoder.init_hidden(self.batch_size)

        out, hidden, cell = decoder(
            self.dec_input,
            hidden,
            cell,
            torch.ones(
                self.batch_size, self.enc_seq_len, self.dec_hidden_size
            ),
        )

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )

    def test_transformer_no_adapter(self):
        """Test if decoder works with same sized encoder states."""
        decoder = TransformerDecoder(
            dec_hidden_size=self.dec_hidden_size,
            enc_hidden_size=self.dec_hidden_size,
            output_size=self.vocab_size,
            num_layers=self.num_layers,
            attention_module=DotProductAttention,
            dec_seq_len=self.dec_seq_len,
            enc_seq_len=self.enc_seq_len,
        )

        out = decoder(
            self.dec_input,
            torch.ones(
                self.batch_size, self.enc_seq_len, self.dec_hidden_size
            ),
        )

        self.assertSequenceEqual(
            out.shape,
            torch.Size([self.batch_size, self.dec_seq_len, self.vocab_size]),
        )


if __name__ == "__main__":
    unittest.main()
