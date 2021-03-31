import torch
import torch.nn.functional as F
import operator
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gen_input(message: str, dataset):
    """Generate encoder input from message."""
    tokens = dataset.get_tokens(message)
    tokens = [
        dataset.vocab[token]
        if token in dataset.vocab
        else dataset.vocab["<unk>"]
        for token in tokens
    ]
    inp = (
        tokens[: dataset.max_length - 1]
        + [dataset.vocab["<new>"]]
        + [dataset.vocab["<pad>"]]
        * max(dataset.max_length - len(tokens) - 1, 0)
    )

    return torch.tensor(inp)


def decode_beam(inp, encoder, decoder, dataset, beam_width):
    """Return the beam_width top predictions of the decoder given an input."""
    # go into eval mode (disable dropout)
    encoder.eval()
    decoder.eval()

    hidden_size = encoder.hidden_size
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
        enc_hidden = enc_hidden.view(3, 2, batch_size, hidden_size)
        enc_hidden = torch.cat(
            (enc_hidden[:, 0], enc_hidden[:, 1]), dim=1
        ).view(3, batch_size, hidden_size * 2)

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

    for pick in top_picks:
        print(
            np.exp(pick["prob"].item()),
            [dataset.inverse_vocab[token.item()] for token in pick["seq"]],
        )
