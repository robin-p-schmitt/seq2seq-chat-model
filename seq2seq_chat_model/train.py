import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from seq2seq_chat_model.dataset import ChatDataset
from seq2seq_chat_model.models.encoders import LSTMEncoder
from seq2seq_chat_model.models.decoders import LSTMAttentionDecoder
import os
from pathlib import Path
import gensim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(encoder, decoder, dataset, epochs=500, batch_size=512):
    """Train Seq2Seq network on a given dataset"""
    # define trainloader
    trainloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # use cross entropy as loss and Adam as optimizer
    criterion = torch.nn.CrossEntropyLoss()
    encoder_opt = torch.optim.Adam(encoder.parameters())
    decoder_opt = torch.optim.Adam(decoder.parameters())

    vocab_size = len(dataset.vocab)
    hidden_size = encoder.hidden_size

    encoder.train()
    decoder.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for i, data in enumerate(trainloader):
            # get data and move it to the device
            input_tensor, output_tensor = data
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            loss = 0

            # init encoder hidden state and cell
            enc_hidden = encoder.init_hidden(batch_size)
            enc_cell = encoder.init_hidden(batch_size)
            # obtain encoder outputs
            enc_outputs, enc_hidden, enc_cell = encoder(
                input_tensor, enc_hidden, enc_cell
            )
            # concatenate the last hidden states from both directions
            enc_hidden = enc_hidden.view(
                encoder.num_layers, 2, batch_size, hidden_size
            )
            enc_hidden = torch.cat(
                (enc_hidden[-1, 0], enc_hidden[-1, 1]), dim=1
            ).view(1, batch_size, hidden_size * 2)

            # init decoder hidden and cell state
            dec_hidden = decoder.init_hidden(batch_size)
            dec_cell = decoder.init_hidden(batch_size)

            # pass the indices from the target sentence into the
            # decoder, one at a time
            for i in range(output_tensor.size(1) - 1):
                dec_in = output_tensor[:, i].view(-1, 1)
                # use teacher forcing
                target = output_tensor[:, i + 1].view(-1, 1)

                # produce next decoder output
                dec_out, dec_hidden, dec_cell = decoder(
                    dec_in, enc_outputs, enc_hidden, dec_hidden, dec_cell
                )

                # add to loss
                loss += criterion(
                    torch.reshape(dec_out, (-1, vocab_size)),
                    torch.reshape(target, (-1,)),
                )

            # do backpropagation and update weights
            loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            # add to running loss
            running_loss += loss.item()

        # print current mean loss after every epoch
        print(
            "Epoch {} - Loss: {}".format(
                epoch,
                running_loss
                / ((len(dataset) // batch_size) * dataset.max_length),
            )
        )
