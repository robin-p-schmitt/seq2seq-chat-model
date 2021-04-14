import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from seq2seq_chat_model.models.utils import positional_encoding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionDecoder(nn.Module, ABC):
    """Base class for all decoders with attention mechanism.

    Attributes:
        dec_hidden_size (int): hidden size to use in decoder
        enc_hidden_size (int): hidden size of the coupled encoder
        output_size (int): vocabulary size to project to
        num_layers (int): number of decoder layers
        attention_module (attention.Attention): attention module to 
            use for attention mechanism
        dec_seq_len (int): length of decoder input sequence
        enc_seq_len (int): length of encoder output sequences
        d_k (int): dimensionality of projected keys/queries. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        d_v (int): dimensionality of projected values. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        n_heads (int): number of attention heads. Defaults to 1.
        pretrained_emb (torch.nn.Embedding): embedding layer to 
            use for initialization. If ``None``, embeddings are
            initialized randomly.
    """

    def __init__(
        self,
        dec_hidden_size,
        enc_hidden_size,
        output_size,
        num_layers,
        attention_module,
        d_k=None,
        d_v=None,
        n_heads=1,
        pretrained_emb=None,
    ):
        super(AttentionDecoder, self).__init__()

        self.dec_hidden_size = dec_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.attention_mod = attention_module
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        # initialize embeddings, if given
        if pretrained_emb is None:
            self.embedding = nn.Embedding(
                output_size, dec_hidden_size, padding_idx=0
            )
        else:
            self.embedding = pretrained_emb

        if dec_hidden_size != enc_hidden_size:
            self.enc_projection = nn.Linear(enc_hidden_size, dec_hidden_size)
        else:
            self.enc_projection = nn.Identity()

    @abstractmethod
    def forward(self):
        ...


class LSTMAttentionDecoder(AttentionDecoder):
    """Decoder for a Seq2Seq network.

    Takes as input the lastly predicted output index and obtains the
    embedding. The embedding then attends over the sequence of encoder
    vectors and produces a context vector. Finally, the embedding and
    the context vector are concatenated and passed to an LSTM network.
    Additionally, the last encoder state is added to the previous
    decoder state in every time step.

    Attributes:
        dec_hidden_size (int): hidden size to use in decoder
        enc_hidden_size (int): hidden size of the coupled encoder
        output_size (int): vocabulary size to project to
        num_layers (int): number of decoder layers
        attention_module (attention.Attention): attention module to 
            use for attention mechanism
        dec_seq_len (int): length of decoder input sequence
        enc_seq_len (int): length of encoder output sequences
        d_k (int): dimensionality of projected keys/queries. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        d_v (int): dimensionality of projected values. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        n_heads (int): number of attention heads. Defaults to 1.
        pretrained_emb (torch.nn.Embedding): embedding layer to 
            use for initialization. If ``None``, embeddings are
            initialized randomly.
    """

    def __init__(
        self,
        dec_hidden_size,
        enc_hidden_size,
        output_size,
        num_layers,
        attention_module,
        d_k=None,
        d_v=None,
        n_heads=1,
        pretrained_emb=None,
    ):
        super().__init__(
            dec_hidden_size,
            enc_hidden_size,
            output_size,
            num_layers,
            attention_module,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            pretrained_emb=pretrained_emb,
        )

        # LSTM network with num_layers layers
        self.lstm = nn.LSTM(
            self.dec_hidden_size + self.enc_hidden_size,
            self.dec_hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.attention = self.attention_mod(
            enc_hidden_size,
            dec_hidden_size,
            d_k,
            d_v,
            n_heads,
            masked=False,
        )

        # last layer which projects decoder state to the size of the
        # output vocabulary
        self.projection = nn.Linear(self.dec_hidden_size, self.output_size)

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

        last_enc_out = self.enc_projection(enc_outputs[None, :, -1])

        # add last encoder output to all previous lstm hidden states
        hidden = hidden + last_enc_out

        # lstm forward pass + projection to vocabulary
        out, (hidden, cell) = self.lstm(combined, [hidden, cell])
        out = self.projection(out)

        return out, hidden, cell

    def init_hidden(self, batch_size):
        # initialize hidden state with zeros
        return torch.zeros(
            self.num_layers, batch_size, self.dec_hidden_size, device=device
        )


class TransformerDecoderBlock(nn.Module):
    """Basic building block for the Transformer decoder.

    Attributes:
        hidden_size (int): hidden size
        ff_size (int): hidden layer size of feed forward net
        attention_module (attention.Attention): attention module to 
            use for attention mechanism
        dec_seq_len (int): length of decoder input sequence
        enc_seq_len (int): length of encoder output sequences
        d_k (int, optional): dimensionality of projected keys/queries. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        d_v (int, optional): dimensionality of projected values. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        n_heads (int, optional): number of attention heads. Defaults to 1.
    """

    def __init__(
        self,
        hidden_size,
        ff_size,
        attention_module,
        d_k = None,
        d_v = None,
        n_heads = 1,
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.hidden_size = hidden_size

        self.attention = attention_module(
            hidden_size,
            hidden_size,
            d_k,
            d_v,
            n_heads,
            masked=False,
        )

        self.masked_attention = attention_module(
            hidden_size,
            hidden_size,
            d_k,
            d_v,
            n_heads,
            masked=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.conv_ff1 = nn.Conv1d(hidden_size, ff_size, 1)
        self.conv_ff2 = nn.Conv1d(ff_size, hidden_size, 1)

    def forward(self, dec_inputs, enc_outputs):
        """Forward pass of the Transformer encoder block.

        Args:
            enc_inputs (torch.tensor): input of shape
                (batch, seq_len, hidden)

        Returns:
            torch.tensor: output of shape
                (batch, seq_len, hidden)
        """

        context = self.masked_attention(*([dec_inputs] * 3))
        res = context + dec_inputs
        res = self.layer_norm(res)
        context = self.attention(*[enc_outputs] * 2, dec_inputs)
        res = context + res
        res = self.layer_norm(res)
        ff = self.conv_ff1(res.view(-1, self.hidden_size, 1))
        ff = self.conv_ff2(F.relu(ff)).view(*res.shape[:2], self.hidden_size)
        res = res + ff
        res = self.layer_norm(res)

        return res


class TransformerDecoder(AttentionDecoder):
    """Implementation of the Transformer encoder as described in:
    https://arxiv.org/pdf/1706.03762.pdf.

    Attributes:
        dec_hidden_size (int): hidden size to use in decoder
        enc_hidden_size (int): hidden size of the coupled encoder
        output_size (int): vocabulary size to project to
        num_layers (int): number of decoder layers
        attention_module (attention.Attention): attention module to 
            use for attention mechanism
        dec_seq_len (int): length of decoder input sequence
        enc_seq_len (int): length of encoder output sequences
        d_k (int, optional): dimensionality of projected keys/queries. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        d_v (int, optional): dimensionality of projected values. Defaults
            to ``dec_hidden_size`` / ``n_heads``.
        n_heads (int, optional): number of attention heads. Defaults to 1.
        ff_size (int, optional): hidden layer size of feed forward net.
            Defaults to ``dec_hidden_size`` * 4.
        pretrained_emb (torch.nn.Embedding, optional): embedding layer to 
            use for initialization. If ``None``, embeddings are
            initialized randomly.
    """

    def __init__(
        self,
        dec_hidden_size,
        enc_hidden_size,
        output_size,
        num_layers,
        attention_module,
        d_k=None,
        d_v=None,
        n_heads=1,
        ff_size=None,
        pretrained_emb=None,
    ):
        super().__init__(
            dec_hidden_size,
            enc_hidden_size,
            output_size,
            num_layers,
            attention_module,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            pretrained_emb=pretrained_emb,
        )

        if ff_size is None:
            ff_size = dec_hidden_size * 4

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    self.dec_hidden_size,
                    ff_size,
                    attention_module,
                    d_k,
                    d_v,
                    n_heads,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.projection = nn.Linear(self.dec_hidden_size, self.output_size)

    def forward(self, dec_inputs, enc_outputs):
        """Forward pass of the Transformer encoder.

        Args:
            enc_inputs (torch.tensor): inputs of shape
                (batch, seq_len)
        Returns:
            torch.tensor: outputs of shape
                (batch, seq_len, output_size)
        """

        embeddings = self.embedding(dec_inputs)
        enc_outputs = self.enc_projection(enc_outputs)
        print(embeddings.shape)
        print(enc_outputs.shape)

        pe = positional_encoding(dec_inputs.shape[1], self.dec_hidden_size)

        outputs = embeddings + pe[None]

        print(outputs.shape)

        for block in self.blocks:
            outputs = block(outputs, enc_outputs)

        outputs = self.projection(outputs)

        return outputs
