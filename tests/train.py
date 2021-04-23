import unittest
from seq2seq_chat_model.models.encoders import TransformerEncoder
from seq2seq_chat_model.models.decoders import TransformerDecoder
from seq2seq_chat_model.models.attention import DotProductAttention
from seq2seq_chat_model.models.utils import (
    get_decoding_trans,
    get_encoding_trans,
    split_dataset,
)
from seq2seq_chat_model.dataset import MockDataset
from seq2seq_chat_model.train import train_step, train_epoch, evaluate_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = MockDataset()

        self.encoder = TransformerEncoder(
            input_size=len(self.dataset.vocab),
            hidden_size=128,
            num_layers=4,
            attention_module=DotProductAttention,
        )
        self.decoder = TransformerDecoder(
            dec_hidden_size=128,
            enc_hidden_size=128,
            output_size=len(self.dataset.vocab),
            num_layers=4,
            attention_module=DotProductAttention,
        )

        self.encoder_opt = Adam(self.encoder.parameters())
        self.decoder_opt = Adam(self.decoder.parameters())
        self.criterion = CrossEntropyLoss()

        train_sampler, val_sampler, test_sampler = split_dataset(
            self.dataset, 0.1, 0.1
        )

        self.train_loader = DataLoader(
            self.dataset, batch_size=32, sampler=train_sampler, drop_last=False
        )

        self.val_loader = DataLoader(
            self.dataset, batch_size=32, sampler=val_sampler, drop_last=False
        )

        self.test_loader = DataLoader(
            self.dataset, batch_size=32, sampler=test_sampler, drop_last=False
        )

    def test_train_step(self):
        input_tensor, target_tensor = next(iter(self.train_loader))
        train_step(
            input_tensor,
            target_tensor,
            self.encoder,
            self.decoder,
            self.criterion,
            get_encoding_trans,
            get_decoding_trans,
        )

    def test_train_epoch(self):
        train_epoch(
            self.encoder,
            self.decoder,
            self.encoder_opt,
            self.decoder_opt,
            self.criterion,
            get_encoding_trans,
            get_decoding_trans,
            self.train_loader,
            self.val_loader,
        )

    def test_evaluate_model(self):
        evaluate_model(
            self.encoder,
            self.decoder,
            get_encoding_trans,
            get_decoding_trans,
            self.test_loader,
            beam_width=3,
            max_n=2,
        )
