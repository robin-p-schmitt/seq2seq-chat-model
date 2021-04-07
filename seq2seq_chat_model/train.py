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
import numpy as np
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using " + str(device) + " as device.")


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
        logging.info(
            "Epoch {} - Loss: {}".format(
                epoch,
                running_loss
                / ((len(dataset) // batch_size) * dataset.max_length),
            )
        )


if __name__ == "__main__":
    # init variables
    data_path = os.path.join("seq2seq_chat_model", "data", "training")
    save_dest = os.path.join("seq2seq_chat_model", "models", "saved")
    if not os.path.exists(save_dest):
        os.mkdir(save_dest)
    log_path = os.path.join(save_dest, "train_log.log")
    hidden_size = 128
    learning_setups = {}
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # init movie dataset
    print("INIT MOVIE DATASET")
    mov_path = Path(data_path).glob("movie-corpus.txt")
    mov_dataset = ChatDataset(mov_path, max_length=20, min_count=10)
    print("LENGTH OF MOVIE DATASET: ", len(mov_dataset))

    # pretrain movie embeddings
    print("Train movie embeddings")
    word2vec = gensim.models.Word2Vec(
        mov_dataset.sentences,
        size=hidden_size,
        window=10,
        min_count=mov_dataset.min_count,
        negative=5,
        iter=30,
    )
    # load pretrained embeddings into dataset and create torch embedding layer
    print("Load movie embeddings")
    mov_dataset.load_word2vec(word2vec)
    mov_embedding = nn.Embedding.from_pretrained(
        (mov_dataset.get_embeddings())
    ).to(device)

    # define encoder and decoder for movie dialogue task
    encoder = LSTMEncoder(
        mov_embedding.num_embeddings, hidden_size, 9, mov_embedding
    ).to(device)
    decoder = LSTMAttentionDecoder(
        hidden_size, mov_embedding.num_embeddings, 7, mov_embedding
    ).to(device)

    # add movie setup to setups
    learning_setups["movie"] = {
        "embedding": mov_embedding,
        "projection": decoder.projection,
        "dataset": mov_dataset,
    }

    # init whatsapp dataset
    print("INIT WHATSAPP DATASET")
    wa_paths = Path(data_path).glob("whatsapp*.txt")
    wa_dataset = ChatDataset(wa_paths, max_length=20, min_count=5)
    print("LENGTH OF WHATSAPP DATASET: ", len(wa_dataset))

    # pretrain whatsapp embeddings
    print("Train whatsapp embeddings")
    word2vec = gensim.models.Word2Vec(
        wa_dataset.sentences,
        size=hidden_size,
        window=7,
        min_count=wa_dataset.min_count,
        negative=5,
        iter=100,
    )

    # load pretrained embeddings into dataset and create torch embedding layer
    print("Load whatsapp embeddings")
    wa_dataset.load_word2vec(word2vec)
    wa_embedding = nn.Embedding.from_pretrained(
        (wa_dataset.get_embeddings())
    ).to(device)

    # add whatsapp setup
    learning_setups["whatsapp"] = {
        "embedding": wa_embedding,
        "projection": nn.Linear(hidden_size, wa_embedding.num_embeddings).to(
            device
        ),
        "dataset": wa_dataset,
    }

    inference_setup = {}
    inference_setup["german"] = {
        "embedding": learning_setups["whatsapp"]["embedding"],
        "projection": learning_setups["whatsapp"]["projection"],
        "vocab": learning_setups["whatsapp"]["dataset"].vocab,
    }
    inference_setup["english"] = {
        "embedding": learning_setups["movie"]["embedding"],
        "projection": learning_setups["movie"]["projection"],
        "vocab": learning_setups["movie"]["dataset"].vocab,
    }

    print("Start Multitask Training")
    for i in range(250):
        # randomly select one of the tasks
        print("Iteration " + str(i))
        logging.info("Iteration " + str(i))
        setup = np.random.choice(list(learning_setups.values()), 1)[0]

        # change embedding and projection layers to selected task
        # the remaining paramerters are shared across tasks
        encoder.embedding = setup["embedding"]
        decoder.embedding = setup["embedding"]
        decoder.projection = setup["projection"]

        # train the current setup for 5 epochs
        train(encoder, decoder, setup["dataset"], epochs=3, batch_size=512)

    print("Start fine-tuning")
    setup = learning_setups["whatsapp"]

    # change embedding and projection layers to selected task
    # the remaining paramerters are shared across tasks
    encoder.embedding = setup["embedding"]
    decoder.embedding = setup["embedding"]
    decoder.projection = setup["projection"]

    # fine-tuning for target task (whatsapp)
    train(encoder, decoder, setup["dataset"], epochs=250, batch_size=512)

    # save data for inference
    inference_setup = {}
    inference_setup["german"] = {
        "embedding": learning_setups["whatsapp"]["embedding"],
        "projection": learning_setups["whatsapp"]["projection"],
        "vocab": learning_setups["whatsapp"]["dataset"].vocab,
    }
    inference_setup["english"] = {
        "embedding": learning_setups["movie"]["embedding"],
        "projection": learning_setups["movie"]["projection"],
        "vocab": learning_setups["movie"]["dataset"].vocab,
    }

    # save setups
    torch.save(inference_setup, os.path.join(save_dest, "setup.pt"))

    # save models
    torch.save(
        encoder,
        os.path.join(save_dest, "encoder.pt"),
    )
    torch.save(
        decoder,
        os.path.join(save_dest, "decoder.pt"),
    )
