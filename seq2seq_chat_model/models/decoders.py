import torch
import torch.nn as nn
from abc import ABC, abstractmethod

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module, ABC):
    """[summary]

    [extended_summary]

    Args:
        nn ([type]): [description]
        ABC ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        enc_hidden_size,
        num_layers,
        attention,
        pretrained_emb=None,
    ):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.enc_hidden_size = enc_hidden_size
        self.num_layers = num_layers
        self.attention = attention

        # initialize embeddings, if given
        if pretrained_emb is None:
            self.embedding = nn.Embedding(
                output_size, hidden_size, padding_idx=0
            )
        else:
            self.embedding = pretrained_emb

        self.enc_projection = nn.Linear(enc_hidden_size, hidden_size)

    @abstractmethod
    def forward(self):
        ...


class LSTMAttentionDecoder(Decoder):
    """Decoder for a Seq2Seq network.

    Takes as input the lastly predicted output index and obtains the
    embedding. The embedding then attends over the sequence of encoder
    vectors and produces a context vector. Finally, the embedding and
    the context vector are concatenated and passed to an LSTM network.
    Additionally, the last encoder state is added to the previous
    decoder state in every time step.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        enc_hidden_size,
        num_layers,
        attention,
        pretrained_emb=None,
    ):
        super().__init__(
            hidden_size,
            output_size,
            enc_hidden_size,
            num_layers,
            attention,
            pretrained_emb,
        )

        # LSTM network with num_layers layers
        self.lstm = nn.LSTM(
            hidden_size + enc_hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.1,
        )

        # last layer which projects decoder state to the size of the
        # output vocabulary
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, dec_inputs, hidden, cell, enc_outputs):
        """Forward pass through the decoder.

        The flow is as follows:
            1. obtain embedding of decoder input
            2. for every decoder embedding, obtain weighted context vectors
            3. concatenate embedding and context vector
            4. add the last encoder output to the previous hidden lstm states
            5. forward pass through lstm
            6. projection to vocabulary

        Args:
            dec_inputs (torch.tensor): decoder inputs of shape
                (batch, dec_seq_len)
            hidden (torch.tensor): decoder hidden states of shape
                (num_dec_layers, batch, dec_hid)
            cell (torch.tensor): decoder cell states of shape
                (num_dec_layers, batch, dec_hid)
            enc_outputs (torch.tensor): encoder outputs of shape
                (batch, enc_seq_len, enc_hid)

        Returns:
            torch.tensor: unnormalized output scores for next token.
            torch.tensor: current hidden state of decoder.
            torch.tensor: current cell state of decoder.
        """
        # obtain embedding from lastly predicted symbol
        # shape: (batch, dec_seq_len, dec_hid)
        embedding = self.embedding(dec_inputs)

        # obtain weighted context vector
        # shape: (batch, dec_seq_len, enc_hid)
        context = self.attention(enc_outputs, enc_outputs, embedding)

        # concatenate embedding and context
        combined = torch.cat([embedding, context], dim=-1)
        # add last encoder output to all previous lstm hidden states
        hidden = hidden + self.enc_projection(enc_outputs[None, :, -1])

        # lstm forward pass + projection to vocabulary
        out, (hidden, cell) = self.lstm(combined, [hidden, cell])
        out = self.projection(out)

        return out, hidden, cell

    def init_hidden(self, batch_size):
        # initialize hidden state with zeros
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
