import torch
import torch.nn as nn
from seq2seq_chat_model.dataset import ChatDataset
from seq2seq_chat_model.models.encoders import LSTMEncoder
from seq2seq_chat_model.models.decoders import LSTMAttentionDecoder
from seq2seq_chat_model.models.utils import decode_beam
import os
from pathlib import Path
import gensim
import numpy as np
import logging
from nltk.translate import bleu_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    criterion,
    encode_func,
    decode_func,
):
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    enc_outputs = encode_func(input_tensor, encoder)

    dec_outputs = decode_func(target_tensor[:, :-1], enc_outputs, decoder)

    loss = criterion(
        dec_outputs.reshape(-1, dec_outputs.shape[-1]),
        target_tensor[:, 1:].reshape(-1),
    )

    return loss


def train_epoch(
    encoder,
    decoder,
    encoder_opt,
    decoder_opt,
    criterion,
    encode_func,
    decode_func,
    train_loader,
    val_loader=None,
):
    """Train Seq2Seq network on a given dataset"""

    encoder.train()
    decoder.train()

    train_loss = 0
    val_loss = None

    for data in train_loader:

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        loss = train_step(
            *data, encoder, decoder, criterion, encode_func, decode_func
        )
        train_loss += loss

        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

    if val_loader is not None:
        encoder.eval()
        decoder.eval()
        val_loss = 0
        for data in val_loader:
            val_loss += train_step(
                *data, encoder, decoder, criterion, encode_func, decode_func
            )

    return train_loss / len(train_loader), val_loss / len(val_loader)


def evaluate_model(
    encoder,
    decoder,
    encode_func,
    decode_func,
    dataloader,
    beam_width=3,
    max_n=2,
):
    encoder.eval()
    decoder.eval()

    bleu = 0

    for data in dataloader:
        input_tensor, output_tensor = map(lambda x: x.to(device), data)

        sos_index = output_tensor[0, 0]
        max_length = output_tensor.shape[1] - 1

        best_sequence = decode_beam(
            input_tensor,
            encoder,
            decoder,
            encode_func,
            decode_func,
            beam_width,
            sos_index,
            max_length,
        )
        best_sequence = best_sequence[0][:, 0]

        reference = output_tensor.unsqueeze(dim=1).tolist()
        candidate = best_sequence.tolist()
        bleu += bleu_score.corpus_bleu(
            reference, candidate, weights=[1 / max_n] * max_n
        )

    return bleu / len(dataloader)


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
