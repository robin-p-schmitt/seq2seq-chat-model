import unittest
import torch
from seq2seq_chat_model.models.encoders import TransformerEncoder, LSTMEncoder
from seq2seq_chat_model.models.decoders import TransformerDecoder, LSTMAttentionDecoder
from seq2seq_chat_model.models.attention import DotProductAttention
from seq2seq_chat_model.models.utils import get_encoding_trans, get_decoding_trans, decode_beam, get_decoding_rnn, get_encoding_rnn

class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 512
        self.vocab_size = 1000
        self.hidden_size = 128

        self.input_len = 10
        self.output_len = 7

        self.input_tensor = torch.ones(self.batch_size, self.input_len).long()
        self.output_tensor = torch.ones(self.batch_size, self.output_len).long()
        self.encoder_outputs = torch.ones(self.batch_size, self.input_len, self.hidden_size)

        self.rnn_encoders = [
            LSTMEncoder(self.vocab_size, self.hidden_size, 4)
        ]

        self.rnn_decoders = [
            LSTMAttentionDecoder(self.hidden_size, self.hidden_size, self.vocab_size, 4, DotProductAttention),
        ]

        self.trans_encoders = [
            TransformerEncoder(self.vocab_size, self.hidden_size, 4, DotProductAttention)
        ]

        self.trans_decoders = [
            TransformerDecoder(self.hidden_size, self.hidden_size, self.vocab_size, 4, DotProductAttention)
        ]

        self.beam_width = 8
        self.num_decoding_steps = 20

    # def test_encoding_rnn(self):
    #     for encoder in self.rnn_encoders:
    #         with self.subTest(encoder = encoder.__class__):
    #             encoding = get_encoding_rnn(self.input_tensor, encoder)

    #             self.assertSequenceEqual(encoding.shape, [self.batch_size, self.input_len, self.hidden_size * 2])

    # def test_encoding_trans(self):
    #     for encoder in self.trans_encoders:
    #         with self.subTest(encoder = encoder.__class__):
    #             encoding = get_encoding_trans(self.input_tensor, encoder)

    #             self.assertSequenceEqual(encoding.shape, [self.batch_size, self.input_len, self.hidden_size])

    # def test_decoding_rnn(self):
    #     for decoder in self.rnn_decoders:
    #         with self.subTest(decoder = decoder.__class__):
    #             decoding = get_decoding_rnn(self.output_tensor, self.encoder_outputs, decoder)

    #             self.assertSequenceEqual(decoding.shape, [self.batch_size, self.output_len, self.vocab_size])

    # def test_decoding_trans(self):
    #     for decoder in self.trans_decoders:
    #         with self.subTest(decoder = decoder.__class__):
    #             decoding = get_decoding_trans(self.output_tensor, self.encoder_outputs, decoder)

    #             self.assertSequenceEqual(decoding.shape, [self.batch_size, self.output_len, self.vocab_size])

    def test_beam_search(self):
        top_seqs, top_probs = decode_beam(self.input_tensor, self.trans_encoders[0], self.trans_decoders[0], get_encoding_trans, get_decoding_trans, self.beam_width, sos_index=2, num_steps=self.num_decoding_steps)

        self.assertSequenceEqual(top_seqs.shape, [self.batch_size, self.beam_width, self.num_decoding_steps])
        self.assertSequenceEqual(top_probs.shape, [self.batch_size, self.beam_width])

if __name__ == "__main__":
    unittest.main()


