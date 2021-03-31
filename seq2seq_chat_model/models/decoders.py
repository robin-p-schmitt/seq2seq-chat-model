import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMAttentionDecoder(nn.Module):
    """Decoder for a Seq2Seq network.

    Takes as input the lastly predicted output index and obtains the
    embedding. The embedding then attends over the sequence of encoder
    vectors and produces a context vector. Finally, the embedding and
    the context vector are concatenated and passed to an LSTM network.
    Additionally, the last encoder state is added to the previous
    decoder state in every time step.
    """

    def __init__(
        self, hidden_size, output_size, num_layers, pretrained_emb=None
    ):
        super(LSTMAttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # initialize embeddings, if given
        if pretrained_emb is None:
            self.embedding = nn.Embedding(
                output_size, hidden_size, padding_idx=0
            )
        else:
            self.embedding = pretrained_emb

        # the weights for the Bahdanau attention mechanism i.e.
        # energy(s_t', h_t) = comb_attn_w * (enc_attn_w * h_t +
        # dec_attn_w * s_t'), where s_t' is the decoder state at time
        # t' and h_t is the encoder state at time t
        self.enc_attn_w = nn.Linear(2 * hidden_size, hidden_size)
        self.dec_attn_w = nn.Linear(hidden_size, hidden_size)
        self.comb_attn_w = nn.Linear(hidden_size, 1)
        # softmax for normalizing attention energies
        self.attn_softmax = nn.Softmax(dim=-1)

        # linear layer for scaling down the last encoder state in
        # order to add it to the previous decoder state (bc the
        # encoder is bidirectional, its states are double the size of
        # decoder states)
        self.scale_enc_hidden = nn.Linear(2 * hidden_size, hidden_size)

        # LSTM network with num_layers layers
        self.lstm = nn.LSTM(
            hidden_size * 3,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.1,
        )

        # last layer which projects decoder state to the size of the
        # output vocabulary
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, y, enc_outputs, enc_hidden, hidden, cell):
        """Forward pass through the decoder.

        Args: y: lastly predicted output with shape (batch, 1)
            enc_outputs: sequence of encoder vectors of shape (batch,
            seq_len, hidden_size * 2) enc_hidden: the last encoder
            output of shape (batch, 1, hidden_size * 2) hidden: the
            last hidden state of the lstm network of shape
            (num_dec_layers, batch, hidden_size) cell: the last cell
            state of the lstm network of shape (num_dec_layers, batch,
            hidden_size)
        """
        # obtain embedding from lastly predicted symbol
        embedding = self.embedding(y)

        # obtain unnormalized attention energies by using Bahdanau
        # attention attn_energies is of shape (batch, 1, seq_len) and
        # contains energies for current decoder state and all encoder
        # states
        attn_energies = torch.tanh(
            self.dec_attn_w(embedding[:, :, None])
            + self.enc_attn_w(enc_outputs[:, None])
        )
        attn_energies = self.comb_attn_w(attn_energies)
        attn_energies = torch.squeeze(attn_energies, dim=-1)

        # obtain attention scores by normalizing energies
        attn_weights = F.softmax(attn_energies, dim=-1)
        # weigth encoder outputs with attention scores
        context = attn_weights[:, :, :, None] * enc_outputs[:, None]
        # obtain weighted sum
        context = torch.sum(context, dim=-2)

        # concatenate context and embedding
        combined = torch.cat((embedding, context), dim=-1)
        # scale last encoder state by a factor of 2
        enc_hidden = self.scale_enc_hidden(enc_hidden)

        # apply lstm network
        out, (hidden, cell) = self.lstm(combined, (hidden + enc_hidden, cell))
        # project to output vocabulary
        out = self.linear(out)
        # remove seq_len dimension
        out = torch.squeeze(out, dim=1)

        return out, hidden, cell

    def init_hidden(self, batch_size):
        # initialize hidden state with zeros
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
