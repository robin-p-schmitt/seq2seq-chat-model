import torch
import torch.nn.functional as F
import operator
import numpy as np
import nltk
from seq2seq_chat_model.data.prepare_data import (
    replace_digits,
    replace_multichars,
)
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tokenize_message(message: str):
    """Clean message and return list of list of strings.

    Splits the message by the special <new> token, applies preprocessing
    steps to the message and tokenizes it into words.

    Args:
        message (str): the message to tokenize
    Returns:
        group (List[List[str]]) : a list of tokenized messages
    """
    message = replace_multichars(message)
    group = message.split("<new>")
    group = [nltk.word_tokenize(msg) for msg in group]
    group = [[replace_digits(token) for token in msg] for msg in group]

    return group


def get_encoder_input(question_group: List[List[str]], dataset):
    # link together sequences of one group with the <new> token e.g.
    # question_group = [["hey"], ["how", "are", "you", "?"]] ->
    # question = ["hey", "<new>", "how", "are", "you", "?"]
    question = [token for seq in question_group for token in seq + ["<new>"]]
    # replace every token with its unique index or with the <unk> index
    # if it is not in the vocabulary
    question = [
        dataset.vocab[token]
        if token in dataset.vocab
        else dataset.vocab["<unk>"]
        for token in question
    ]

    # either cut off long sequences or pad short sequences so that
    # every sequence has length max_length
    question = question[: dataset.max_length] + [dataset.vocab["<pad>"]] * max(
        dataset.max_length - len(question), 0
    )

    return torch.tensor(question)


def get_decoder_input(answer_group: List[List[str]], dataset):
    # link together sequences of one group with the <new> token e.g.
    # question_group = [["hey"], ["how", "are", "you", "?"]] ->
    # question = ["hey", "<new>", "how", "are", "you", "?"]
    answer = [token for seq in answer_group for token in seq + ["<new>"]]
    # replace every token with its unique index or with the <unk> index
    # if it is not in the vocabulary
    answer = [
        dataset.vocab[token]
        if token in dataset.vocab
        else dataset.vocab["<unk>"]
        for token in answer
    ]

    # additionally, add sos and eos tokens to start and end of the
    # answer
    answer = (
        [dataset.vocab["<start>"]]
        + answer[: dataset.max_length - 2]
        + [dataset.vocab["<stop>"]]
        + [dataset.vocab["<pad>"]]
        * max(dataset.max_length - len(answer) - 2, 0)
    )

    return torch.tensor(answer)


def decode_beam(inp, encoder, decoder, dataset, beam_width):
    """Return the beam_width top predictions of the decoder given an input."""
    # go into eval mode (disable dropout)
    encoder.eval()
    decoder.eval()

    hidden_size = encoder.hidden_size
    num_enc_lay = encoder.num_layers
    batch_size = 1

    top_picks = []

    with torch.no_grad():
        # init encoder inputs
        enc_hidden = encoder.init_hidden(batch_size)
        enc_cell = encoder.init_hidden(batch_size)
        enc_in = inp.view(batch_size, -1).to(device)

        # obtain encoder outputs
        enc_outputs, enc_hidden, enc_cell = encoder(
            enc_in, enc_hidden, enc_cell
        )

        # prepare hidden encoder state for decoder
        enc_hidden = enc_hidden.view(num_enc_lay, 2, batch_size, hidden_size)
        enc_hidden = torch.cat(
            (enc_hidden[-1, 0], enc_hidden[-1, 1]), dim=1
        ).view(1, batch_size, hidden_size * 2)

        # init decoder inputs
        dec_hidden = decoder.init_hidden(batch_size)
        dec_cell = decoder.init_hidden(batch_size)
        dec_in = torch.tensor(dataset.vocab["<start>"]).view(1, 1).to(device)

        # obtain first decoder outputs
        dec_out, dec_hidden, dec_cell = decoder(
            dec_in, enc_outputs, enc_hidden, dec_hidden, dec_cell
        )

        # get first top predictions
        top_k = torch.topk(F.softmax(dec_out, 1), beam_width, 1)

        # save parameters of the first top picks
        top_picks = [
            {
                "seq": [token],
                "prob": np.log(prob.item()),
                "hid": dec_hidden,
                "cell": dec_cell,
            }
            for prob, token in zip(top_k[0][0], top_k[1][0])
        ]

        # do 10 decoding steps
        for i in range(10):
            hypotheses = []

            # go through every current top pick
            for pick in top_picks:
                # the lastly predicted symbol is the next input
                dec_in = pick["seq"][-1].view(1, 1)

                # get the next outputs
                dec_out, dec_hidden, dec_cell = decoder(
                    dec_in, enc_outputs, enc_hidden, pick["hid"], pick["cell"]
                )

                # get next top picks of the current hypothesis
                top_k = torch.topk(F.softmax(dec_out, 1), beam_width, 1)

                # store parameters of the top picks
                picks = [
                    {
                        "seq": pick["seq"] + [token],
                        "prob": np.log(prob.item()) + pick["prob"],
                        "hid": dec_hidden,
                        "cell": dec_cell,
                    }
                    for prob, token in zip(top_k[0][0], top_k[1][0])
                ]

                # add to current hypothesis
                hypotheses += picks

            # sort after probability
            hypotheses = sorted(
                hypotheses, key=operator.itemgetter("prob"), reverse=True
            )

            # get top k hyptheses
            top_picks = hypotheses[:beam_width]

    top_probs = [np.exp(pick["prob"].item()) for pick in top_picks]
    top_seqs = [pick["seq"] for pick in top_picks]

    return list(zip(top_probs, top_seqs))


def clean_dec_output(sequence):
    sequence = [
        token for token in sequence if token != "<pad>" and token != "<stop>"
    ]
    sequence = " ".join(sequence)
    sequence = sequence.replace("<new>", "\n")

    return sequence


def positional_encoding(seq_len, hidden_size):
    """Implementation of the positional encoding proposed in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    # positions and dimensions
    positions = torch.arange(0, seq_len)
    dimensions = torch.arange(0, hidden_size, dtype=torch.float)

    # the divisor term of the pe
    div_term = torch.pow(10000, 2 * dimensions / hidden_size)
    # the term before applying the sinusoids
    inner = positions[:, None] / div_term[None, :]
    # init pe with zeros
    pe = torch.zeros(seq_len, hidden_size)
    # apply sin/cos to all even/uneven dimensions
    pe[:, 0::2] = torch.sin(inner[:, 0::2])
    pe[:, 1::2] = torch.cos(inner[:, 1::2])

    return pe


def attention_mask(seq_len):
    scores = torch.zeros(seq_len, seq_len)
    for i in range(1, seq_len):
        scores = scores + torch.diag(
            torch.tensor([float("-inf")] * (seq_len - i)), i
        )

    return scores
