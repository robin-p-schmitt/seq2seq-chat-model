from seq2seq_chat_model.models.encoders import LSTMEncoder
from seq2seq_chat_model.models.decoders import LSTMAttentionDecoder
from seq2seq_chat_model.dataset import ChatDataset
from seq2seq_chat_model.models.utils import tokenize_message, get_encoder_input, decode_beam
import os
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    version_dir = os.path.join("seq2seq_chat_model", "models", "saved")
    versions = os.listdir(version_dir)

    print("\n"*10 + "+"*6 + " Willkommen zum ChatBot " + "+"*6)
    print("\n\n")
    
    selected_version = None
    while(selected_version is None):
        sys_msg = "Bitte wähle eine der folgenden Versionen: \n\n"
        sys_msg += "\n".join(versions) + "\n\n"
        sys_msg += "Eingabe: "
        selected_version = input(sys_msg)

        if not os.path.exists(os.path.join(version_dir, selected_version)):
            print("\nBitte wähle eine gültige Version!")
            selected_version = None

    selected_path = os.path.join(version_dir, selected_version)
    enc_state_dict = torch.load(os.path.join(selected_path, "encoder.pt"), map_location=device)
    dec_state_dict = torch.load(os.path.join(selected_path, "decoder.pt"), map_location=device)
    setup = torch.load(os.path.join(selected_path, "setups.pt"), map_location=device)
    dataset = setup["dataset"]

    encoder = LSTMEncoder(
        len(dataset.vocab), 128, 7
    ).to(device)
    decoder = LSTMAttentionDecoder(
        128, len(dataset.vocab), 5
    ).to(device)

    encoder.load_state_dict(enc_state_dict)
    decoder.load_state_dict(dec_state_dict)

    username = input("Gib einen Nutzernamen ein: ")

    for i in range(10):
        message = input("\n\n" + username + ": ")
 
        message = tokenize_message(message)
        message = get_encoder_input(message, dataset)

        answer = decode_beam(message, encoder, decoder, dataset, 5)[0][1]
        print("Bot: ", answer)


