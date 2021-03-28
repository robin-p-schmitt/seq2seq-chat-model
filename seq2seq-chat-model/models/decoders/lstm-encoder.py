import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """Encoder for a Seq2Seq network.

    Takes a sequence of word indices as input and obtains the
    embeddings. The embeddings are then passed through a bi-LSTM
    network to produce a sequence of encoder vectors.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, pretrained_emb=None
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # initialize embeddings, if given
        if pretrained_emb is None:
            self.embedding = nn.Embedding(
                input_size, hidden_size, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb)

        # bi-LSTM network with num_layers layers and dropout
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

    def forward(self, x, hidden, cell):
        """Forward pass of the encoder.

        Args: x: sequence of word indices of shape (batch, seq_len)
            hidden: last hidden state of the bi-LSTM of shape (2 *
            num_enc_layers, batch, hidden_size) cell: last cell state
            of the bi-LSTM of shape (2 * num_enc_layers, batch,
            hidden_size)
        """
        # obtain embedding of input index
        embedding = self.embedding(x)
        # get the encoder outputs
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        return output, hidden, cell

    def init_hidden(self, batch_size):
        # initialize hidden state with zeros
        return torch.zeros(
            2 * self.num_layers, batch_size, self.hidden_size, device=device
        )
