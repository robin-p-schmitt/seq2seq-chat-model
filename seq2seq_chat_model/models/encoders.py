import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from seq2seq_chat_model.models.utils import positional_encoding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module, ABC):
    """Base class for all encoders.
    
    Attributes:
        input_size (int): vocabulary size
        hidden_size (int): hidden size
        num_layers (int): number of encoder layers
        pretrained_emb (torch.nn.Module, optional): initializes the
            embedding layer with the given layer
    """

    def __init__(
        self, input_size, hidden_size, num_layers, pretrained_emb=None
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # initialize embeddings, if given
        if pretrained_emb is None:
            self.embedding = nn.Embedding(
                input_size, hidden_size, padding_idx=0
            )
        else:
            self.embedding = pretrained_emb

    @abstractmethod
    def forward(self):
        ...


class LSTMEncoder(Encoder):
    """Encoder for a Seq2Seq network.

    Takes a sequence of word indices as input and obtains the
    embeddings. The embeddings are then passed through a bi-LSTM
    network to produce a sequence of encoder vectors.

    Attributes:
        input_size (int): vocabulary size
        hidden_size (int): hidden size
        num_layers (int): number of encoder layers
        pretrained_emb (torch.nn.Module, optional): initializes the
            embedding layer with the given layer
    """

    def __init__(
        self, input_size, hidden_size, num_layers, pretrained_emb=None
    ):
        super().__init__(
            input_size, hidden_size, num_layers, pretrained_emb=pretrained_emb
        )

        # bi-LSTM network with num_layers layers and dropout
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

    def forward(self, enc_inputs, hidden, cell):
        """Forward pass of the encoder.

        Args:
            enc_inputs (torch.tensor): inputs of shape
                (batch, seq_len)
            hidden (torch.tensor): the hidden states to pass to the
                lstm network of shape (num_layers * 2, batch, hidden)
            cell (torch.tensor): the cell states to pass to the lstm
                network of shape (num_layers * 2, batch, hidden)

        Returns:
            torch.tensor: output of shape (batch, seq_len, hidden * 2)
            torch.tensor: hidden states of shape
                (num_layers * 2, batch, hidden)
            torch.tensor: cell states of shape
                (num_layers * 2, batch, hidden)

        """
        # obtain embedding of input index
        embedding = self.embedding(enc_inputs)
        # get the encoder outputs
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        return output, hidden, cell

    def init_hidden(self, batch_size):
        # initialize hidden state with zeros
        return torch.zeros(
            2 * self.num_layers, batch_size, self.hidden_size, device=device
        )


class TransformerEncoderBlock(nn.Module):
    """Building block for the Transformer encoder.

    Attributes:
        hidden_size (int): hidden size
        seq_len (int): length of the input sequences
        attention_module (models.attention.Attention): attention
            module reference to use for self-attention.
        ff_size (int): size of the two-layer feed forward net
        d_k (int): dimensionality of projected keys/queries
        d_v (int): dimensionality of projected values
        n_heads (int): number of attention heads used
    """
    def __init__(
        self,
        hidden_size,
        attention_module,
        ff_size,
        d_k,
        d_v,
        n_heads,
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.hidden_size = hidden_size

        self.attention = attention_module(
            hidden_size,
            hidden_size,
            d_k,
            d_v,
            n_heads,
            masked=False,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.conv_ff1 = nn.Conv1d(hidden_size, ff_size, 1)
        self.conv_ff2 = nn.Conv1d(ff_size, hidden_size, 1)

    def forward(self, enc_inputs):
        """Forward pass of the Transformer encoder block.

        Args:
            enc_inputs (torch.tensor): input of shape
                (batch, seq_len, hidden)

        Returns:
            torch.tensor: output of shape
                (batch, seq_len, hidden)
        """

        context = self.attention(*[enc_inputs] * 3)
        res = context + enc_inputs
        res = self.layer_norm(res)
        ff = self.conv_ff1(res.view(-1, self.hidden_size, 1))
        ff = self.conv_ff2(F.relu(ff)).view(*res.shape[:2], self.hidden_size)
        res = res + ff
        res = self.layer_norm(res)

        return res


class TransformerEncoder(Encoder):
    """Implementation of the Transformer encoder as described in:
    https://arxiv.org/pdf/1706.03762.pdf.

    Attributes:
        input_size (int): vocabulary size
        hidden_size (int): hidden size
        num_layers (int): number of encoder layers
        seq_len (int): length of the input sequences
        attention_module (models.attention.Attention): attention
            module reference to use for self-attention.
        ff_size (int, optional): size of the two-layer feed forward
            net. Defaults to ``hidden_size`` * 4. 
        d_k (int, optional): dimensionality of projected keys/queries.
            Defaults to ``hidden_size`` / ``n_heads``.
        d_v (int, optional): dimensionality of projected values. 
            Defaults to ``hidden_size`` / ``n_heads``.
        n_heads (int, optional): number of attention heads used.
            Defaults to 1.
        pretrained_emb (torch.nn.Module, optional): initializes the
            embedding layer with the given layer. If ``None``,
            embedding is initialized randomly.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        attention_module,
        ff_size=None,
        d_k=None,
        d_v=None,
        n_heads=1,
        pretrained_emb=None,
    ):
        super().__init__(
            input_size, hidden_size, num_layers, pretrained_emb=pretrained_emb
        )

        if ff_size is None:
            ff_size = hidden_size * 4

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size,
                    attention_module,
                    ff_size,
                    d_k,
                    d_v,
                    n_heads,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, enc_inputs):
        """Forward pass of the Transformer encoder.

        Args:
            enc_inputs (torch.tensor): inputs of shape
                (batch, seq_len)
        Returns:
            torch.tensor: outputs of shape
                (batch, seq_len, hidden)
        """

        embeddings = self.embedding(enc_inputs)
        
        pe = positional_encoding(enc_inputs.shape[1], self.hidden_size)
        outputs = embeddings + pe[None]

        for block in self.blocks:
            outputs = block(outputs)

        return outputs
