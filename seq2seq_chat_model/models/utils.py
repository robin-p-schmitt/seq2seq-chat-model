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

def get_encoding_rnn(input_tensor, encoder):
    batch_size = input_tensor.shape[0]
    hidden = cell = encoder.init_hidden(batch_size)

    out, _, _ = encoder(input_tensor, hidden, cell)

    return out

def get_encoding_trans(input_tensor, encoder):
    return encoder(input_tensor)

def get_decoding_rnn(target_tensor, enc_outputs, decoder):
    batch_size = target_tensor.shape[0]
    dec_hidden = dec_cell = decoder.init_hidden(batch_size)

    dec_outputs = []

    for i in range(target_tensor.shape[1]):
        dec_input = target_tensor[:, i].view(-1, 1)

        dec_output, dec_hidden, dec_cell = decoder(
            dec_input, dec_hidden, dec_cell, enc_outputs
        )

        dec_outputs.append(dec_output)

    return torch.cat(dec_outputs, dim = 1)

def get_decoding_trans(target_tensor, enc_outputs, decoder):
    return decoder(target_tensor, enc_outputs)

def decode_beam(input_tensor, encoder, decoder, encoding_func, decoding_func, beam_width, sos_index, num_steps):
    """Return the beam_width top predictions of the decoder given an input."""
    batch_size = input_tensor.shape[0]

    with torch.no_grad():
        # get encoder sequence
        enc_outputs = encoding_func(input_tensor, encoder)
        # first decoder input is the sos token (seq len = 1)
        dec_in = torch.tensor([sos_index] * batch_size).view(batch_size, 1).to(device)
        # get the decoder predictions
        dec_out = decoding_func(dec_in, enc_outputs, decoder)

        # get first top predictions
        top_k = torch.topk(F.softmax(dec_out, dim = -1), k = beam_width, dim = -1)

        # top len 1 sequences of shape (batch, beam_width, 1)
        top_seqs = top_k[1].transpose(1, 2)
        # corresponding log probs of shape (batch, beam_width)
        top_log_probs =  torch.log(top_k[0]).squeeze()
        print(top_seqs.shape)
        print(top_log_probs.shape)
                 

        # do num_steps decoding steps
        for i in range(num_steps):

            # merge batch and beam width dimensions
            # shape (batch * beam_width, seq_len)
            dec_in = top_seqs.view(batch_size * beam_width, -1)
            # pass the current top sequences to the decoder
            dec_out = decoding_func(dec_in, enc_outputs, decoder)
 
            # for every batch and beam, get the next top k predictions
            probs, tokens = torch.topk(F.softmax(dec_out[:, -1], dim = -1), k = beam_width, dim = -1)
            # separate batch and beam dimension and merge the two beam dimensions instead
            # (for every previous beam, there are now k new beams)
            tokens = tokens.view(batch_size, beam_width * beam_width, 1)
            # for every previous beam, repeat the corresponding sequence for beam_width times
            seqs = top_seqs.repeat_interleave(repeats = beam_width, dim = 1)
            # append the new beam predictions to the corresponding previous beam sequence
            seqs = torch.cat([seqs, tokens], dim = -1)

            # separate batch and previous beam dimension
            # (for every previous beam prob, there are now k new beam probs)
            probs = probs.view(batch_size, beam_width, beam_width)
            # add the log probs of the previous beams to the corresponding log probs
            # of the new beams (corresponds to the products of the non-log probs)
            log_probs = top_log_probs[:, :, None] + torch.log(probs)
            # merge the beam dimensions
            log_probs = log_probs.view(batch_size, beam_width * beam_width)
            # get the new k top beams
            top_log_probs, top_indices = torch.topk(log_probs, k = beam_width, dim = -1)
            
            # select the new top sequences as the top indices of the probabilities
            top_seqs = seqs[torch.arange(seqs.shape[0]), top_indices],

    top_probs = torch.exp(top_log_probs)

    return top_seqs, top_probs


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
