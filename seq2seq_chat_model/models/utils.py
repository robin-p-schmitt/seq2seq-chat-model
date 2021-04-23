import torch
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
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


def get_encoder_input(
    question_group: List[List[str]],
    vocab,
    max_length,
    new_token,
    unk_token,
    pad_token,
):
    """Transform messages in natural language into encoder-readable input.

    Individual messages are connected using the <new> token to symbolize
    the end of an individual message. The tokens are then replaced by
    their corresponding index in the vocabulary. Lastly, the message
    is either trimmed or padded to match the given max length.

    Args:
        question_group (List[List[str]]): messages are given as lists of strings
            and multiple consecutive messages of the same person are given as lists
            of lists of strings.
        vocab (dict): vocabulary which maps string tokens to unique indices.
        max_length (int): the length of the returned messages.
        new_token (str): string used as new token to indicated the end of an
            individual message.
        unk_token (str): string used as unknown token.
        pad_token (str): string used as padding token.

    Returns:
        torch.tensor: 1d tensor containing the indices of the tokens of the message.
    """
    # link together sequences of one group with the <new> token e.g.
    # question_group = [["hey"], ["how", "are", "you", "?"]] ->
    # question = ["hey", "<new>", "how", "are", "you", "?"]
    question = [token for seq in question_group for token in seq + [new_token]]
    # replace every token with its unique index or with the <unk> index
    # if it is not in the vocabulary
    question = [
        vocab[token] if token in vocab else vocab[unk_token]
        for token in question
    ]

    # either cut off long sequences or pad short sequences so that
    # every sequence has length max_length
    question = question[:max_length] + [vocab[pad_token]] * max(
        max_length - len(question), 0
    )

    return torch.tensor(question)


def get_decoder_input(
    answer_group: List[List[str]],
    vocab,
    max_length,
    new_token,
    unk_token,
    pad_token,
    sos_token,
    eos_token,
):
    """Transform messages in natural language into decoder-readable input.

    Individual messages are connected using the <new> token to symbolize
    the end of an individual message. The tokens are then replaced by
    their corresponding index in the vocabulary. Lastly, the message
    is either trimmed or padded to match the given max length and its further
    extended with a start-of-sequence and an end-of-sequence token.

    Args:
        question_group (List[List[str]]): messages are given as lists of strings
            and multiple consecutive messages of the same person are given as lists
            of lists of strings.
        vocab (dict): vocabulary which maps string tokens to unique indices.
        max_length (int): the length of the returned messages.
        new_token (str): string used as new token to indicated the end of an
            individual message.
        unk_token (str): string used as unknown token.
        pad_token (str): string used as padding token.
        sos_token (str): string used as start-of-sequence token.
        eos_token (str): string used as end-of-sequence token.

    Returns:
        torch.tensor: 1d tensor containing the indices of the tokens of the message.
    """
    # link together sequences of one group with the <new> token e.g.
    # question_group = [["hey"], ["how", "are", "you", "?"]] ->
    # question = ["hey", "<new>", "how", "are", "you", "?"]
    answer = [token for seq in answer_group for token in seq + [new_token]]
    # replace every token with its unique index or with the <unk> index
    # if it is not in the vocabulary
    answer = [
        vocab[token] if token in vocab else vocab[unk_token]
        for token in answer
    ]

    # additionally, add sos and eos tokens to start and end of the
    # answer
    answer = (
        [vocab[sos_token]]
        + answer[: max_length - 2]
        + [vocab[eos_token]]
        + [vocab[pad_token]] * max(max_length - len(answer) - 2, 0)
    )

    return torch.tensor(answer)


def get_encoding_rnn(input_tensor, encoder):
    """Given an input, get encoder sequence from recurrent encoder.

    Args:
        input_tensor (torch.tensor): input of shape (batch_size, seq_len)
        encoder (torch.module): recurrent encoder which expects and returns
            three tensors (in/out, hidden state, cell state)

    Returns:
        torch.tensor: output of shape (batch_size, seq_len, hidden_size)
    """
    batch_size = input_tensor.shape[0]
    hidden = cell = encoder.init_hidden(batch_size)

    out, _, _ = encoder(input_tensor, hidden, cell)

    return out


def get_encoding_trans(input_tensor, encoder):
    """Given an input, get encoder sequence from Transformer-like encoder.

    Args:
        input_tensor (torch.tensor): input of shape (batch_size, seq_len)
        encoder (torch.module): Transformer-like encoder which expects
        and returns only one tensor (input and output)

    Returns:
        torch.tensor: output of shape (batch_size, seq_len, hidden_size)
    """
    return encoder(input_tensor)


def get_decoding_rnn(target_tensor, enc_outputs, decoder):
    """Given an input, get decoder sequence from recurrent decoder.

    Args:
        target_tensor (torch.tensor): input of shape
            (batch_size, seq_len)
        decoder (torch.module): recurrent decoder which expects 4 inputs
            (previous outputs, hidden, cell, enc_outputs) and produces
            3 outputs (out, hidden, cell)

    Returns:
        torch.tensor: output of shape (batch_size, seq_len, vocab_size)
    """
    batch_size = target_tensor.shape[0]
    dec_hidden = dec_cell = decoder.init_hidden(batch_size)

    dec_outputs = []

    for i in range(target_tensor.shape[1]):
        dec_input = target_tensor[:, i].view(-1, 1)

        dec_output, dec_hidden, dec_cell = decoder(
            dec_input, dec_hidden, dec_cell, enc_outputs
        )

        dec_outputs.append(dec_output)

    return torch.cat(dec_outputs, dim=1)


def get_decoding_trans(target_tensor, enc_outputs, decoder):
    """Given an input, get decoder sequence from Transformer-like decoder.

    Args:
        target_tensor (torch.tensor): input of shape
            (batch_size, seq_len)
        decoder (torch.module): recurrent decoder which expects 2 inputs
            (previous outputs, enc_outputs) and produces
            1 output (output)

    Returns:
        torch.tensor: output of shape (batch_size, seq_len, vocab_size)
    """
    return decoder(target_tensor, enc_outputs)


def split_dataset(dataset, test_frac, val_frac, seed=None):
    """Returns two samplers that split the dataset at the given percentile.

    The returned samplers are shuffled.

    Args:
        dataset (torch.utils.data.Dataset): the dataset to split.
        percent (float): the percentile at which to split.
        seed (int, optional): the random seed to use for reproducability.
    """

    dataset_size = len(dataset)
    indices = np.arange(start=0, stop=dataset_size)
    train_test_split = int(test_frac * dataset_size)
    train_val_split = int((test_frac + val_frac) * dataset_size)

    if seed:
        np.random.seed(seed)
    np.random.shuffle(indices)

    test_split = indices[:train_test_split]
    val_split = indices[train_test_split:train_val_split]
    train_split = indices[train_val_split:]

    test_sampler = SubsetRandomSampler(test_split)
    val_sampler = SubsetRandomSampler(val_split)
    train_sampler = SubsetRandomSampler(train_split)

    return train_sampler, val_sampler, test_sampler


def decode_beam(
    input_tensor,
    encoder,
    decoder,
    encoding_func,
    decoding_func,
    beam_width,
    sos_index,
    num_steps,
):
    """[summary]

    [extended_summary]

    Args:
        input_tensor ([type]): [description]
        encoder ([type]): [description]
        decoder ([type]): [description]
        encoding_func ([type]): [description]
        decoding_func ([type]): [description]
        beam_width ([type]): [description]
        sos_index ([type]): [description]
        num_steps ([type]): [description]

    Returns:
        [type]: [description]
    """
    batch_size = input_tensor.shape[0]

    with torch.no_grad():
        # get encoder sequence
        enc_outputs = encoding_func(input_tensor, encoder)
        # first decoder input is the sos token (seq len = 1)
        dec_in = (
            torch.tensor([sos_index] * batch_size)
            .view(batch_size, 1)
            .to(device)
        )
        # get the decoder predictions
        dec_out = decoding_func(dec_in, enc_outputs, decoder)

        # get first top predictions
        top_k = torch.topk(F.softmax(dec_out, dim=-1), k=beam_width, dim=-1)

        # top len 1 sequences of shape (batch, beam_width, 1)
        top_seqs = top_k[1].transpose(1, 2)
        # corresponding log probs of shape (batch, beam_width)
        top_log_probs = torch.log(top_k[0]).squeeze(dim=1)

        # encoder outputs stay the same across beams
        enc_outputs = enc_outputs.repeat_interleave(repeats=beam_width, dim=0)

        # do num_steps decoding steps
        for i in range(num_steps - 1):

            # merge batch and beam width dimensions
            # shape (batch * beam_width, seq_len)
            dec_in = top_seqs.view(batch_size * beam_width, -1)
            # pass the current top sequences to the decoder
            dec_out = decoding_func(dec_in, enc_outputs, decoder)

            # for every batch and beam, get the next top k predictions
            probs, tokens = torch.topk(
                F.softmax(dec_out[:, -1], dim=-1), k=beam_width, dim=-1
            )
            # separate batch and beam dimension and merge the two beam dimensions instead
            # (for every previous beam, there are now k new beams)
            tokens = tokens.view(batch_size, beam_width * beam_width, 1)
            # for every previous beam, repeat the corresponding sequence for beam_width times
            seqs = top_seqs.repeat_interleave(repeats=beam_width, dim=1)
            # append the new beam predictions to the corresponding previous beam sequence
            seqs = torch.cat([seqs, tokens], dim=-1)

            # separate batch and previous beam dimension
            # (for every previous beam prob, there are now k new beam probs)
            probs = probs.view(batch_size, beam_width, beam_width)
            # add the log probs of the previous beams to the corresponding log probs
            # of the new beams (corresponds to the products of the non-log probs)
            log_probs = top_log_probs[:, :, None] + torch.log(probs)
            # merge the beam dimensions
            log_probs = log_probs.view(batch_size, beam_width * beam_width)
            # get the new k top beams
            top_log_probs, top_indices = torch.topk(
                log_probs, k=beam_width, dim=-1
            )
            # expand indices to match the sequence dimensions
            top_indices = top_indices[:, :, None].expand(
                -1, -1, seqs.shape[-1]
            )

            # select the new top sequences as the top indices of the probabilities
            top_seqs = seqs.gather(dim=1, index=top_indices)

    # convert log probs to probs
    top_probs = torch.exp(top_log_probs)

    return top_seqs, top_probs


def format_dec_output(sequence, vocab):
    """Format decoder output into readable, individual messages.

    Removes <pad> and <stop> tokens and splits by <new> tokens.

    Args:
        sequence (List[torch.tensor(int)]): decoder output indices.

    Returns:
        List[str]: list of individual string sentences.
    """

    sequence = [vocab[token.item()] for token in sequence]
    sequence = [
        token for token in sequence if token != "<pad>" and token != "<stop>"
    ]
    sequence = " ".join(sequence)
    sequence = sequence.split("<new>")
    sequence = list(map(str.strip, sequence))

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
    """Return simple attention mask.

    Returns a 0 zero matrix with all values above the
    main diagonal set to negative infinity.
    When used for masked attention, this means that each
    token only attends to tokens up to itself and not to
    future tokens.

    Args:
        seq_len (int): dimension of returned matrix.

    Returns:
        torch.tensor: attention mask.
    """
    scores = torch.zeros(seq_len, seq_len)
    for i in range(1, seq_len):
        # add diagonal of -inf to the zero matrix
        scores = scores + torch.diag(
            torch.tensor([float("-inf")] * (seq_len - i)), i
        )

    return scores
